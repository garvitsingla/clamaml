import os
import warnings
import logging
warnings.filterwarnings("ignore") 
logging.getLogger("gymnasium").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.multiprocessing as mp
from functools import partial
import numpy as np
import torch
import gc
import time
import os
import json
import matplotlib.pyplot as plt
import random
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.metalearners.lang_trpo_hyper import MAMLTRPO
import sampler_lang as S
from sampler_lang import (BabyAIMissionTaskWrapper, 
                        MissionEncoder,
                        SentenceMissionEncoder,
                        MissionParamAdapter,
                        ConstraintParamAdapter,
                        MultiTaskSampler, 
                        preprocess_obs)
from environment import (LOCAL_MISSIONS,
                         DOOR_MISSIONS,
                         OPEN_DOOR_MISSIONS,
                         DOOR_LOC_MISSIONS,
                         PICKUP_MISSIONS,
                         OPEN_DOORS_ORDER_MISSIONS,
                         ACTIONOBJDOOR_MISSIONS,
                         FINDOBJS5_MISSIONS,
                         CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
                         CONSTRAINED_FINDOBJS5_MISSIONS,
                         CONSTRAINED_LOCAL_MISSIONS,
                         CONSTRAINT_TEXTS,
                         DOUBLE_CONSTRAINT_TEXTS,
                         PICKUP_MISSIONS,
                         CONSTRAINED_PICKUP_MISSIONS,
                         DOUBLE_CONSTRAINED_LOCAL_MISSIONS,
                         DOUBLE_CONSTRAINED_PICKUP_MISSIONS,
                         DOUBLE_CONSTRAINED_GOTOOPEN_MISSIONS,
                         DOUBLE_CONSTRAINED_OPENDOOR_MISSIONS,
                         DOUBLE_CONSTRAINED_OPENDOORLOC_MISSIONS,
                         DOUBLE_CONSTRAINED_OPENDOORSORDER_MISSIONS,
                         DOUBLE_CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
                         DOUBLE_CONSTRAINED_FINDOBJS5_MISSIONS)
from environment import (GoToLocalMissionEnv,
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ConstrainedGoToLocalEnv,
                         ConstrainedPickupDistEnv,
                         ConstrainedGoToObjDoorEnv,
                         ConstrainedGoToOpenEnv,
                         ConstrainedOpenDoorEnv,
                         ConstrainedOpenDoorLocEnv,
                         ConstrainedOpenDoorsOrderEnv,
                         ConstrainedActionObjDoorEnv,
                         ConstrainedFindObjS5Env)
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparser
p = argparse.ArgumentParser()
p.add_argument("--env", dest="env_name",
               choices=["GoToLocal","PickupDist","GoToObjDoor","GoToOpen","OpenDoor",
                        "OpenDoorLoc","OpenDoorsOrder",
                        "ConstrainedGoToLocal","ConstrainedPickupDist",
                        "ConstrainedGoToObjDoor","ConstrainedGoToOpen",
                        "ConstrainedOpenDoor","ConstrainedOpenDoorLoc",
                        "ConstrainedOpenDoorsOrder", "ConstrainedActionObjDoor",
                        "ConstrainedFindObjS5"],
               default="ConstrainedGoToLocal")
p.add_argument("--room-size", type=int, default=8)
p.add_argument("--num-dists", type=int, default=2)
p.add_argument("--max-steps", type=int, default=300)
p.add_argument("--delta-theta", type=float, default=0.3)
p.add_argument("--meta-iters", type=int, default=200, help="number of meta-batches")
p.add_argument("--batch-size", type=int, default=40, help="episodes per meta-batch (per task)")
p.add_argument("--num-workers", type=int, default=4)
p.add_argument("--lambda-lava", type=float, default=0.8)
p.add_argument("--lambda-grass", type=float, default=0.3)
p.add_argument("--lambda-water", type=float, default=0.5)
p.add_argument("--hazard-density", type=float, default=0.2)
p.add_argument("--max-hazards", type=int, default=2,
               help="max number of hazard tiles placed per constraint type")
p.add_argument("--delta-constraint", type=float, default=0.1,
               help="scale for constraint adapter deltas (like delta-theta but for constraints)")

args = p.parse_args()


# Build the environment
def build_env(env, room_size, num_dists, max_steps, missions, hazard_density=0.2,
              goals=None, constraints=None, max_hazards=2):

    if env == "GoToLocal":
        base = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "PickupDist":
        base = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "GoToObjDoor":
        base = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
    elif env == "GoToOpen":
        base = GoToOpenMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "OpenDoor":
        base = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorLoc":
        base = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorsOrder":
        base = OpenDoorsOrderMissionEnv(room_size=room_size)
    elif env == "ConstrainedGoToLocal":
        base = ConstrainedGoToLocalEnv(room_size=room_size, num_dists=num_dists,
                                       max_steps=max_steps, hazard_density=hazard_density,
                                       max_hazards=max_hazards)
    elif env == "ConstrainedPickupDist":
        base = ConstrainedPickupDistEnv(room_size=room_size, num_dists=num_dists,
                                        max_steps=max_steps, hazard_density=hazard_density,
                                        max_hazards=max_hazards)
    elif env == "ConstrainedGoToObjDoor":
        base = ConstrainedGoToObjDoorEnv(max_steps=max_steps, num_distractors=num_dists,
                                         hazard_density=hazard_density,
                                         max_hazards=max_hazards)
    elif env == "ConstrainedGoToOpen":
        base = ConstrainedGoToOpenEnv(room_size=room_size, num_dists=num_dists,
                                      max_steps=max_steps, hazard_density=hazard_density,
                                      max_hazards=max_hazards)
    elif env == "ConstrainedOpenDoor":
        base = ConstrainedOpenDoorEnv(room_size=room_size, max_steps=max_steps,
                                      hazard_density=hazard_density,
                                      max_hazards=max_hazards)
    elif env == "ConstrainedOpenDoorLoc":
        base = ConstrainedOpenDoorLocEnv(room_size=room_size, max_steps=max_steps,
                                         hazard_density=hazard_density,
                                         max_hazards=max_hazards)
    elif env == "ConstrainedOpenDoorsOrder":
        base = ConstrainedOpenDoorsOrderEnv(room_size=room_size, max_steps=max_steps,
                                            hazard_density=hazard_density,
                                            max_hazards=max_hazards)
    elif env == "ConstrainedActionObjDoor":
        base = ConstrainedActionObjDoorEnv(room_size=room_size, max_steps=max_steps, hazard_density=hazard_density,
                                           max_hazards=max_hazards)
    elif env == "ConstrainedFindObjS5":
        base = ConstrainedFindObjS5Env(room_size=5, max_steps=max_steps, hazard_density=hazard_density,
                                       max_hazards=max_hazards)
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions, goals=goals, constraints=constraints)


# Select for missions based on environment
def select_missions(env_name):
    from environment import (
        CONSTRAINED_GOTOOBJDOOR_MISSIONS, CONSTRAINED_GOTOOPEN_MISSIONS,
        CONSTRAINED_OPENDOOR_MISSIONS, CONSTRAINED_OPENDOORLOC_MISSIONS,
        CONSTRAINED_OPENDOORSORDER_MISSIONS
    )
    mission_map = {
        "ConstrainedGoToLocal": DOUBLE_CONSTRAINED_LOCAL_MISSIONS,
        "ConstrainedPickupDist": DOUBLE_CONSTRAINED_PICKUP_MISSIONS,
        "ConstrainedGoToObjDoor": CONSTRAINED_GOTOOBJDOOR_MISSIONS,
        "ConstrainedGoToOpen": DOUBLE_CONSTRAINED_GOTOOPEN_MISSIONS,
        "ConstrainedOpenDoor": DOUBLE_CONSTRAINED_OPENDOOR_MISSIONS,
        "ConstrainedOpenDoorLoc": DOUBLE_CONSTRAINED_OPENDOORLOC_MISSIONS,
        "ConstrainedOpenDoorsOrder": DOUBLE_CONSTRAINED_OPENDOORSORDER_MISSIONS,
        "ConstrainedActionObjDoor": DOUBLE_CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
        "ConstrainedFindObjS5": DOUBLE_CONSTRAINED_FINDOBJS5_MISSIONS
    }
    return mission_map[env_name]


def main():


    def set_seed(seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = 1
    set_seed(seed)

    env_name  = args.env_name
    room_size = args.room_size
    num_dists = args.num_dists
    max_steps = args.max_steps
    delta_theta = args.delta_theta
    delta_constraint = args.delta_constraint
    num_workers = args.num_workers
    num_batches = args.meta_iters
    batch_size = args.batch_size
    hazard_density = args.hazard_density
    max_hazards = args.max_hazards


    missions = select_missions(env_name)

    # For constrained envs, pass separate goals/constraints
    _CONSTRAINED_GOALS = {
        "ConstrainedGoToLocal":      LOCAL_MISSIONS,
        "ConstrainedPickupDist":     PICKUP_MISSIONS,
        "ConstrainedGoToObjDoor":    LOCAL_MISSIONS + DOOR_MISSIONS,
        "ConstrainedGoToOpen":       LOCAL_MISSIONS,
        "ConstrainedOpenDoor":       OPEN_DOOR_MISSIONS,
        "ConstrainedOpenDoorLoc":    OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS,
        "ConstrainedOpenDoorsOrder": OPEN_DOORS_ORDER_MISSIONS,
        "ConstrainedActionObjDoor":  ACTIONOBJDOOR_MISSIONS,
        "ConstrainedFindObjS5":      FINDOBJS5_MISSIONS,
    }
    if env_name in _CONSTRAINED_GOALS:
        goals_list = _CONSTRAINED_GOALS[env_name]
        constraints_list = DOUBLE_CONSTRAINT_TEXTS
    else:
        goals_list = None
        constraints_list = None

    make_env = partial(
        build_env,
        env_name,
        room_size,
        num_dists,
        max_steps,
        missions,
        hazard_density,
        goals_list,
        constraints_list,
        max_hazards
    )

    env = make_env()
    print(f"Using environment: {env_name}\n"
          f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}  "
          f"delta_theta: {delta_theta}")

    # Policy setup 
    hidden_sizes = (64, 64)
    nonlinearity = torch.nn.functional.tanh

    mission_encoder = SentenceMissionEncoder(
        model_name="all-MiniLM-L6-v2",
        frozen=True,          
        normalize=True,         
        cache=True,           
        device=device
    )
    mission_encoder_output_dim = mission_encoder.output_dim
    mission_adapter_input_dimension = mission_encoder_output_dim

    # Policy Parameters shape
    obs, _ = env.reset()
    vec = preprocess_obs(obs)
    input_size = vec.shape[0]
    output_size = env.action_space.n


    policy = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    policy.share_memory()
    baseline = LinearFeatureBaseline(input_size).to(device)

    policy_param_shapes = [p.shape for p in policy.parameters()]

    from sampler_lang import AbsoluteHyperNetwork
    hypernetwork = AbsoluteHyperNetwork(mission_encoder_output_dim, policy_param_shapes).to(device)

    
    sampler = MultiTaskSampler(
        env=env,
        env_fn=make_env,
        batch_size=batch_size,     
        policy=policy,
        baseline=baseline,
        seed=1,
        num_workers=num_workers
    )

    meta_learner = MAMLTRPO(
        policy=policy,
        mission_encoder=mission_encoder,
        hypernetwork=hypernetwork,
        fast_lr=1e-4,
        first_order=True,
        device=device,
        lambda_weights={2: args.lambda_lava, 3: args.lambda_grass, 4: args.lambda_water}
    )

    # Training loop
    avg_steps_per_batch = []
    std_steps_per_batch = []
    # For constrained env, count tasks from goals × constraints; otherwise from missions
    if goals_list is not None:
        total_tasks = len(goals_list) * len(constraints_list)
    else:
        total_tasks = len(env.missions)
    meta_batch_size = globals().get("meta_batch_size") or min(10, total_tasks)

    start_time = time.time()

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        valid_episodes, step_counts = sampler.sample(
            meta_batch_size,
            meta_learner,
            gamma=0.99,
            gae_lambda=1.0,
            device=device
        )
        
        avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
        avg_steps_per_episode = avg_steps / sampler.batch_size 
        avg_steps_per_batch.append(avg_steps_per_episode)
        std_steps = np.std([s / sampler.batch_size for s in step_counts]) if len(step_counts) > 0 else 0.0
        std_steps_per_batch.append(std_steps)
        print(f"Average steps in Meta-batch {batch+1}: {avg_steps_per_episode}")
        # Log average cost across all episodes in this meta-batch
        total_cost = 0
        count = 0
        for ep in valid_episodes:
            if hasattr(ep, '_costs') and ep._costs is not None:
                total_cost += ep._costs.sum().item()
                count += ep._costs.shape[1]  # batch_size
            elif hasattr(ep, 'costs'):
                try:
                    total_cost += ep.costs.sum().item()
                    count += ep.costs.shape[1]
                except Exception:
                    pass
        avg_cost = total_cost / max(count, 1)
        print(f"Average cost in Meta-batch {batch+1}: {avg_cost:.4f}")

        # print("--- Per-Task Episode Breakdown ---")
        # for i, ep in enumerate(valid_episodes):
        #     mission_str = f"{ep.mission[0]} and {ep.mission[1]}" if isinstance(ep.mission, tuple) else ep.mission
        #     print(f"Task {i+1}: {mission_str}")
        #     if hasattr(ep, 'episode_stats') and ep.episode_stats:
        #         for e_idx, stat in enumerate(ep.episode_stats):
        #             tiles_str = ", ".join([f"{k}:{v}" for k, v in stat['tiles_hit'].items()]) if stat['tiles_hit'] else "None"
        #             print(f"  Ep {e_idx+1} | steps={stat['steps']} | violations={stat['violations']} | hazards_hit={tiles_str}")
        # print("----------------------------------\n")

        meta_learner.step(valid_episodes,valid_episodes)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    end_time = time.time()
    training_time = end_time - start_time
    time_per_iteration = training_time / num_batches
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per iteration: {time_per_iteration:.2f} seconds")

    # Save the trained meta-policy parameters
    os.makedirs("hyper_model", exist_ok=True)
    save_dict = {
        "policy": policy.state_dict(),
        "mission_encoder": mission_encoder.state_dict(),
        "hypernetwork": hypernetwork.state_dict() if hypernetwork else None,
    }
    torch.save(save_dict, f"hyper_model/lang_{env_name}_{delta_theta}.pth")


    # plot
    env_dir = os.path.join("metrics", env_name)
    os.makedirs(env_dir, exist_ok=True) 

    np.save(os.path.join(env_dir, f"la_maml_avg_steps_{delta_theta}.npy"), np.array(avg_steps_per_batch))
    np.save(os.path.join(env_dir, f"la_maml_std_steps_{delta_theta}.npy"), np.array(std_steps_per_batch))
    with open(os.path.join(env_dir, f"la_maml_meta_{delta_theta}.json"), "w") as f:
        json.dump({"label" : "LA-MAML", "env" : env_name}, f)
    
    plt.plot(avg_steps_per_batch)
    plt.xlabel("Meta-batch")
    plt.ylabel("Average steps per episode")
    plt.title(f"Average steps per episode per meta-batch (delta_theta={delta_theta})")
    plt.savefig(os.path.join(env_dir, f"la_maml_plot_{delta_theta}.png"))
    plt.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()