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
import json
import matplotlib.pyplot as plt
import random
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.metalearners.lang_trpo_unified import MAMLTRPO  
import sampler_lang as S
from sampler_lang import (BabyAIMissionTaskWrapper,
                          SentenceMissionEncoder,
                          MissionParamAdapter,
                          MultiTaskSampler,
                          preprocess_obs)
from environment import (LOCAL_MISSIONS,
                         PICKUP_MISSIONS,
                         DOOR_MISSIONS,
                         OPEN_DOOR_MISSIONS,
                         DOOR_LOC_MISSIONS,
                         OPEN_DOORS_ORDER_MISSIONS,
                         ACTIONOBJDOOR_MISSIONS,
                         FINDOBJS5_MISSIONS,
                         CONSTRAINED_LOCAL_MISSIONS,
                         CONSTRAINED_PICKUP_MISSIONS,
                         CONSTRAINED_GOTOOBJDOOR_MISSIONS,
                         CONSTRAINED_OPENDOOR_MISSIONS,
                         CONSTRAINED_OPENDOORLOC_MISSIONS,
                         CONSTRAINED_OPENDOORSORDER_MISSIONS,
                         CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
                         CONSTRAINED_GOTOOPEN_MISSIONS,
                         CONSTRAINED_FINDOBJS5_MISSIONS,
                         DOUBLE_CONSTRAINED_LOCAL_MISSIONS,
                         DOUBLE_CONSTRAINED_PICKUP_MISSIONS,
                         DOUBLE_CONSTRAINED_GOTOOBJDOOR_MISSIONS
                         DOUBLE_CONSTRAINED_OPENDOOR_MISSIONS,
                         DOUBLE_CONSTRAINED_OPENDOORLOC_MISSIONS,
                         DOUBLE_CONSTRAINED_OPENDOORSORDER_MISSIONS,
                         DOUBLE_CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
                         DOUBLE_CONSTRAINED_GOTOOPEN_MISSIONS,
                         DOUBLE_CONSTRAINED_FINDOBJS5_MISSIONS,
                         CONSTRAINT_TEXTS,
                         DOUBLE_CONSTRAINT_TEXTS)
from environment import (
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
               choices=["ConstrainedGoToLocal","ConstrainedPickupDist","ConstrainedGoToObjDoor",
                        "ConstrainedOpenDoor","ConstrainedOpenDoorLoc","ConstrainedOpenDoorsOrder",
                        "ConstrainedActionObjDoor","ConstrainedGoToOpen","ConstrainedFindObjS5"],
               default="ConstrainedGoToLocal")
p.add_argument("--room-size", type=int, default=8)
p.add_argument("--num-dists", type=int, default=2)
p.add_argument("--max-steps", type=int, default=300)
p.add_argument("--delta-theta", type=float, default=0.3)
p.add_argument("--delta-constraint", type=float, default=0.1)
p.add_argument("--meta-iters", type=int, default=200)
p.add_argument("--batch-size", type=int, default=40, help="episodes per meta-batch (per task)")
p.add_argument("--num-workers", type=int, default=4)
p.add_argument("--lambda-lava", type=float, default=0.8)
p.add_argument("--lambda-grass", type=float, default=0.3)
p.add_argument("--lambda-water", type=float, default=0.5)
p.add_argument("--hazard-density", type=float, default=0.2)
p.add_argument("--max-hazards", type=int, default=4, help="max constraint tiles")
p.add_argument("--num-constraints", type=int, default=1)

args = p.parse_args()


# Build the environment
def build_env(env, room_size, num_dists, max_steps, missions, hazard_density=0.2,
              goals=None, constraints=None, max_hazards=2):
    if env == "ConstrainedGoToLocal":
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
        base = ConstrainedActionObjDoorEnv(room_size=room_size, max_steps=max_steps, 
                                           hazard_density=hazard_density,
                                           max_hazards=max_hazards)
    elif env == "ConstrainedGoToOpen":
        base = ConstrainedGoToOpenEnv(room_size=room_size, num_dists=num_dists,
                                      max_steps=max_steps, hazard_density=hazard_density,
                                      max_hazards=max_hazards)
    elif env == "ConstrainedFindObjS5":
        base = ConstrainedFindObjS5Env(room_size=room_size, max_steps=max_steps,
                                       hazard_density=hazard_density,
                                       max_hazards=max_hazards)
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions, goals=goals, constraints=constraints)


# Select for missions based on environment
def select_missions(env_name, num_constraints=1):
    if num_constraints == 1:
        mission_map = {
            "ConstrainedGoToLocal": CONSTRAINED_LOCAL_MISSIONS,
            "ConstrainedPickupDist": CONSTRAINED_PICKUP_MISSIONS,
            "ConstrainedGoToObjDoor": CONSTRAINED_GOTOOBJDOOR_MISSIONS,
            "ConstrainedOpenDoor": CONSTRAINED_OPENDOOR_MISSIONS,
            "ConstrainedOpenDoorLoc": CONSTRAINED_OPENDOORLOC_MISSIONS,
            "ConstrainedOpenDoorsOrder": CONSTRAINED_OPENDOORSORDER_MISSIONS,
            "ConstrainedActionObjDoor": CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
            "ConstrainedGoToOpen": CONSTRAINED_GOTOOPEN_MISSIONS,
            "ConstrainedFindObjS5": CONSTRAINED_FINDOBJS5_MISSIONS
        }
    else:
        mission_map = {
            "ConstrainedGoToLocal": DOUBLE_CONSTRAINED_LOCAL_MISSIONS,
            "ConstrainedPickupDist": DOUBLE_CONSTRAINED_PICKUP_MISSIONS,
            "ConstrainedGoToObjDoor": DOUBLE_CONSTRAINED_GOTOOBJDOOR_MISSIONS,
            "ConstrainedOpenDoor": DOUBLE_CONSTRAINED_OPENDOOR_MISSIONS,
            "ConstrainedOpenDoorLoc": DOUBLE_CONSTRAINED_OPENDOORLOC_MISSIONS,
            "ConstrainedOpenDoorsOrder": DOUBLE_CONSTRAINED_OPENDOORSORDER_MISSIONS,
            "ConstrainedActionObjDoor": DOUBLE_CONSTRAINED_ACTIONOBJDOOR_MISSIONS,
            "ConstrainedGoToOpen": DOUBLE_CONSTRAINED_GOTOOPEN_MISSIONS,
            "ConstrainedFindObjS5": DOUBLE_CONSTRAINED_FINDOBJS5_MISSIONS
        }
    return mission_map[env_name]



def main():
    def set_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(1)

    env_name     = args.env_name
    room_size    = args.room_size
    num_dists    = args.num_dists
    max_steps    = args.max_steps
    delta_theta  = args.delta_theta
    num_workers  = args.num_workers
    num_batches  = args.meta_iters
    batch_size   = args.batch_size
    hazard_density = args.hazard_density
    max_hazards = args.max_hazards

    missions = select_missions(env_name, num_constraints=args.num_constraints)

    CONSTRAINED_GOALS = {
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
    goals_list = CONSTRAINED_GOALS[env_name]
    constraints_list = DOUBLE_CONSTRAINT_TEXTS if args.num_constraints == 2 else CONSTRAINT_TEXTS

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
    print("[Unified LA-MAML]\n" f"Using environment: {env_name}\n"
        f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}\n"
        f"delta_theta: {delta_theta}")

    # Policy setup 
    hidden_sizes  = (64, 64)
    nonlinearity  = torch.nn.functional.tanh

    mission_encoder = SentenceMissionEncoder(
        model_name="all-MiniLM-L6-v2",
        frozen=True,
        normalize=True,
        cache=True,
        device=device
    )
    mission_encoder_output_dim = mission_encoder.output_dim

    obs, _ = env.reset()
    vec = preprocess_obs(obs)
    input_size  = vec.shape[0]
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

    mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)

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
        mission_adapter=mission_adapter,  
        delta_theta=delta_theta,
        fast_lr=1e-4,
        first_order=True,
        device=device,
        lambda_weights={2: args.lambda_lava, 3: args.lambda_grass, 4: args.lambda_water}
    )


    avg_steps_per_batch = []
    std_steps_per_batch = []
    avg_costs_per_batch = []
    std_costs_per_batch = []

    total_tasks    = len(missions)
    meta_batch_size = min(10, total_tasks)

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

        total_cost = 0
        count = 0
        all_costs = []
        for ep in valid_episodes:
            if hasattr(ep, '_costs') and ep._costs is not None:
                total_cost += ep._costs.sum().item()
                count      += ep._costs.shape[1]
                all_costs.extend(ep._costs.sum(dim=0).detach().cpu().numpy())
            elif hasattr(ep, 'costs'):
                try:
                    total_cost += ep.costs.sum().item()
                    count      += ep.costs.shape[1]
                    all_costs.extend(ep.costs.sum(dim=0).detach().cpu().numpy())
                except Exception:
                    pass
        avg_cost = total_cost / max(count, 1)
        std_cost = float(np.std(all_costs)) if len(all_costs) > 0 else 0.0
        avg_costs_per_batch.append(avg_cost)
        std_costs_per_batch.append(std_cost)
        print(f"Average cost in Batch {batch+1}: {avg_cost:.4f}")

        meta_learner.step(valid_episodes, valid_episodes)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    training_time = end_time - start_time
    time_per_iteration = training_time / num_batches
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per iteration: {time_per_iteration:.2f} seconds")

    # Save the trained meta-policy parameters
    os.makedirs("unified_model", exist_ok=True)
    save_dict = {
        "policy": policy.state_dict(),
        "mission_encoder": mission_encoder.state_dict(),
        "mission_adapter": mission_adapter.state_dict(),
    }
    torch.save(save_dict, f"unified_model/lang_{env_name}_{delta_theta}_{args.num_constraints}c.pth")

    # Plotting
    env_dir = os.path.join("metrics", f"{env_name}_{args.num_constraints}c")
    os.makedirs(env_dir, exist_ok=True)
    np.save(os.path.join(env_dir, f"unified_avg_steps_{delta_theta}.npy"), np.array(avg_steps_per_batch))
    np.save(os.path.join(env_dir, f"unified_std_steps_{delta_theta}.npy"), np.array(std_steps_per_batch))
    np.save(os.path.join(env_dir, f"unified_avg_costs_{delta_theta}.npy"), np.array(avg_costs_per_batch))
    np.save(os.path.join(env_dir, f"unified_std_costs_{delta_theta}.npy"), np.array(std_costs_per_batch))
    with open(os.path.join(env_dir, f"unified_meta_{delta_theta}.json"), "w") as f:
        json.dump({"label": "Unified LA-MAML", "env": env_name}, f)

    plt.plot(avg_steps_per_batch)
    plt.xlabel("Meta-batch")
    plt.ylabel("Avg steps per episode")
    plt.title(f"[Unified LA-MAML] {env_name} delta_theta={delta_theta} ({args.num_constraints}c)")
    plt.savefig(os.path.join(env_dir, f"unified_plot_{delta_theta}_{args.num_constraints}c.png"))
    plt.close()

    plt.plot(avg_costs_per_batch)
    plt.xlabel("Meta-batch")
    plt.ylabel("Avg cost per episode")
    plt.title(f"[Unified LA-MAML] {env_name} cost  delta={delta_theta} ({args.num_constraints}c)")
    plt.savefig(os.path.join(env_dir, f"unified_cost_plot_{delta_theta}_{args.num_constraints}c.png"))
    plt.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
