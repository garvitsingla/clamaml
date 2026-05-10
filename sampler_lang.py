import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym
from maml_rl.episode import BatchEpisodes
from sentence_transformers import SentenceTransformer
# import logging

import builtins, io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def silence_sampling_rejected():
    real_print = builtins.print
    buf = io.StringIO()
    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("Sampling rejected: unreachable object"):
            return
        return real_print(*args, **kwargs)
    builtins.print = filtered_print
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            yield
    finally:
        builtins.print = real_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = None
mission_encoder = None

def rollout_one_task(args):
    (make_env_fn, mission, policy_cls, policy_kwargs, 
     policy_state_dict,adapted_params_cpu, batch_size, gamma, lambda_weights) = args
    
    import warnings
    import logging
    warnings.filterwarnings("ignore") # Silence everything in workers
    logging.getLogger("gym").setLevel(logging.ERROR) # Silence gym specific prints

    env = make_env_fn()
    env.reset_task(mission)

    policy = policy_cls(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    obs_list, action_list, reward_list, cost_list, episode_list = [], [], [], [], []
    total_steps = 0
    episode_stats = []  # per-episode: {steps, violations, tiles_hit}

    TILE_NAMES = {2: 'lava', 3: 'grass', 4: 'water'}

    for episode in range(batch_size):
        with silence_sampling_rejected():
            obs, info = env.reset()
        done = False; steps = 0; ep_violations = 0; tiles_hit = {}
        while not done:
            obs_vec = preprocess_obs(obs)
            with torch.no_grad():
                pi = policy(torch.from_numpy(obs_vec[None, :]).float(), 
                            params=adapted_params_cpu)
                action = pi.sample().item()
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info.get('cost', 0.0)
            tile_idx = info.get('tile_index', 0)
            if cost > 0:
                cost = cost * lambda_weights.get(tile_idx, 1.0)
                ep_violations += 1
                name = TILE_NAMES.get(tile_idx, f'tile_{tile_idx}')
                tiles_hit[name] = tiles_hit.get(name, 0) + 1
            done = terminated or truncated
            steps += 1
            obs_list.append(obs_vec)
            action_list.append(action)
            reward_list.append(reward)
            cost_list.append(cost)
            episode_list.append(episode)
        total_steps += steps
        episode_stats.append({'steps': steps, 'violations': ep_violations, 'tiles_hit': tiles_hit})
        
        mission_str = f"{mission[0]} and {mission[1]}" if isinstance(mission, tuple) else str(mission)
        tiles_str = ", ".join([f"{k}:{v}" for k, v in tiles_hit.items()]) if tiles_hit else "None"
        # print(f"Task: '{mission_str}' |  Ep: {episode+1}/{batch_size}  | steps: {steps} | violations: {ep_violations} | hazards_hit: {tiles_str}", flush=True)

    return (mission, total_steps, obs_list, action_list, reward_list, cost_list, episode_list, episode_stats)

# Mission Wrapper
class BabyAIMissionTaskWrapper(gym.Wrapper):
    def __init__(self, env, missions=None, goals=None, constraints=None):
        super().__init__(env)
        # Support both old-style (single mission list) and new-style (goal + constraint lists)
        self.missions = missions  # for non-constrained envs
        self.goals = goals        # for constrained envs
        self.constraints = constraints  # for constrained envs
        self.current_mission = None

    def sample_tasks(self, n_tasks):
        if self.goals is not None and self.constraints is not None:
            # Constrained env: sample (goal, constraint) tuples
            goals = list(np.random.choice(self.goals, n_tasks, replace=True))
            constraints = list(np.random.choice(self.constraints, n_tasks, replace=True))
            return list(zip(goals, constraints))
        else:
            return list(np.random.choice(self.missions, n_tasks, replace=False))

    def reset_task(self, mission):
        if isinstance(mission, tuple):
            goal, constraint = mission
            combined = f"{goal} and {constraint}"
            self.current_mission = combined
        else:
            self.current_mission = mission
        if hasattr(self.env, 'set_forced_mission'):
            self.env.set_forced_mission(self.current_mission)

    def reset(self, **kwargs):        
        result = super().reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        if self.current_mission is not None:
            obs['mission'] = self.current_mission
            if hasattr(self.env, 'unwrapped'):
                self.env.unwrapped.mission = self.current_mission
        if isinstance(result, tuple):
            return obs, info
        else:
            return obs
        

# Mission Encoder
class MissionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class SentenceMissionEncoder(nn.Module):
    """
    Mission text -> sentence embedding (SBERT / SentenceTransformer)

    This is "frozen" by default: the pretrained backbone does NOT get updated.
    Your MissionParamAdapter and policy still learn normally.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", frozen=True, normalize=True, cache=True, device=None):
        super().__init__()
        self.normalize = normalize
        self.cache = cache
        self._cache = {}  # mission_str -> embedding tensor on CPU (to save GPU mem)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # SentenceTransformer is itself a torch module
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.output_dim = self.model.get_sentence_embedding_dimension()

        if frozen:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, missions):
        # missions: str or list[str]
        if isinstance(missions, str):
            missions = [missions]

        # simple caching helps because missions repeat a lot
        out = []
        to_encode = []
        idxs = []
        for i, m in enumerate(missions):
            if self.cache and m in self._cache:
                out.append(self._cache[m].to(self.device))
            else:
                out.append(None)
                to_encode.append(m)
                idxs.append(i)

        if len(to_encode) > 0:
            with torch.no_grad():
                emb = self.model.encode(
                    to_encode,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize
                )  # [k, d] on self.device (because SentenceTransformer got device)

            # fill + cache (store cache on CPU)
            for j, i in enumerate(idxs):
                e = emb[j]
                out[i] = e
                if self.cache:
                    self._cache[to_encode[j]] = e.detach().cpu()

        return torch.stack(out, dim=0)  # [B, d]
        

# MissionParamAdapter 
class MissionParamAdapter(nn.Module):
    def __init__(self, mission_adapter_input_dim, policy_param_shapes):
        super().__init__()
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        self.net = nn.Sequential(
            nn.Linear(mission_adapter_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
            nn.Tanh()  
        )
    def forward(self, mission_emb):
        
        out = self.net(mission_emb)  
        split_points = []
        total = 0
        for shape in self.policy_param_shapes:
            num = torch.Size(shape).numel()
            split_points.append(total + num)
            total += num
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped 


# Constraint Param Adapter (separate learnable weights for constraint path)
class ConstraintParamAdapter(nn.Module):
    """constraint embedding → Δθ_constraint. Same architecture as MissionParamAdapter."""
    def __init__(self, constraint_adapter_input_dim, policy_param_shapes):
        super().__init__()
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        self.net = nn.Sequential(
            nn.Linear(constraint_adapter_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
            nn.Tanh()
        )

    def forward(self, constraint_emb):
        out = self.net(constraint_emb)
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped 
        

class AbsoluteHyperNetwork(nn.Module):
    """
    Absolute Hypernetwork: [flattened_theta, goal_emb, constraint_emb] -> absolute theta_prime
    """
    def __init__(self, encoder_output_dim, policy_param_shapes):
        super().__init__()
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        
        # total params + goal_emb + constraint_emb
        self.input_dim = total_params + (encoder_output_dim * 2)
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_params)
        )
        
    def forward(self, combined_input_tensor):
        out = self.net(combined_input_tensor)
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped 
     

def preprocess_obs(obs):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    
    return np.concatenate([image, direction])
    

# Sampler
class MultiTaskSampler(object):
    def __init__(self,
                 env=None,   
                 env_fn=None,      
                 batch_size=None,        
                 policy=None,
                 baseline=None,
                 seed=None,
                 num_workers=0):   
        assert env is not None, "Must pass BabyAI env"
        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, meta_learner, gamma=0.95, gae_lambda=1.0, device='cpu'):

        tasks = self.sample_tasks(meta_batch_size)  
        all_step_counts = []
        valid_episodes_all = []
        if (self.num_workers or 0) > 0:
            assert self.env_fn is not None, "env_fn no there "
 
            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            tasks = self.sample_tasks(meta_batch_size)

            # compute theta' per task
            adapted_params_cpu = []
            for t in tasks:
                theta_prime = meta_learner.adapt_one(t)
                adapted_params_cpu.append({k: v.detach().cpu() for k, v in theta_prime.items()})

            lambda_weights = getattr(meta_learner, 'lambda_weights', {2: 0.8, 3: 0.3, 4: 0.5})

            worker_args = []
            for t, p in zip(tasks, adapted_params_cpu):
                worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                    policy_state_dict_cpu, p, self.batch_size, gamma, lambda_weights))

            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(rollout_one_task, worker_args))

            valid_episodes_all, all_step_counts = [], []
            for (mission, step_count, obs_list, action_list, reward_list, cost_list, episode_list, episode_stats) in results:
                batch_episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                # Store the original task tuple/string
                batch_episodes.mission = mission  # (goal, constraint) tuple or string
                batch_episodes.episode_stats = episode_stats
                # Parse constraint tiles from mission
                HAZARD_TYPES = {'lava': 2, 'grass': 3, 'water': 4}
                constraint_tiles = []
                mission_str = mission[1] if isinstance(mission, tuple) else mission
                for hazard, idx in HAZARD_TYPES.items():
                    if f"avoid {hazard}" in str(mission_str):
                        constraint_tiles.append(idx)
                batch_episodes.constraint_tiles = constraint_tiles
                for obs, action, reward, cost, episode in zip(obs_list, action_list, reward_list, cost_list, episode_list):
                    batch_episodes.append([obs], [np.array(action)], [np.array(reward)], [np.array(cost)], [np.array(episode)])
                self.baseline.fit(batch_episodes)
                batch_episodes.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                batch_episodes.compute_cost_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(batch_episodes)
                all_step_counts.append(step_count)

            return (valid_episodes_all, all_step_counts)
        else:
            # Single-threaded fallback (num_workers=0)
            assert self.env_fn is not None, "env_fn not provided"

            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            adapted_params_cpu = []
            for t in tasks:
                theta_prime = meta_learner.adapt_one(t)
                adapted_params_cpu.append({k: v.detach().cpu() for k, v in theta_prime.items()})

            lambda_weights = getattr(meta_learner, 'lambda_weights', {2: 0.8, 3: 0.3, 4: 0.5})

            valid_episodes_all, all_step_counts = [], []
            for t, p in zip(tasks, adapted_params_cpu):
                args = (self.env_fn, t, policy_cls, policy_kwargs,
                        policy_state_dict_cpu, p, self.batch_size, gamma, lambda_weights)
                mission, step_count, obs_list, action_list, reward_list, cost_list, episode_list, episode_stats = rollout_one_task(args)

                batch_episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                batch_episodes.mission = mission
                batch_episodes.episode_stats = episode_stats
                HAZARD_TYPES = {'lava': 2, 'grass': 3, 'water': 4}
                constraint_tiles = []
                mission_str = mission[1] if isinstance(mission, tuple) else mission
                for hazard, idx in HAZARD_TYPES.items():
                    if f"avoid {hazard}" in str(mission_str):
                        constraint_tiles.append(idx)
                batch_episodes.constraint_tiles = constraint_tiles
                for obs, action, reward, cost, episode in zip(obs_list, action_list, reward_list, cost_list, episode_list):
                    batch_episodes.append([obs], [np.array(action)], [np.array(reward)], [np.array(cost)], [np.array(episode)])
                self.baseline.fit(batch_episodes)
                batch_episodes.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                batch_episodes.compute_cost_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(batch_episodes)
                all_step_counts.append(step_count)

            return (valid_episodes_all, all_step_counts)