import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from collections import OrderedDict


class MAMLTRPO(GradientBasedMetaLearner):

    def __init__(self,  
                 policy,
                 mission_encoder,
                 mission_adapter,
                 constraint_adapter=None,
                 fast_lr=0.5,
                 delta_theta=None,
                 delta_constraint=None,
                 first_order=False,
                 device='cpu',
                 lambda_weights=None):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.delta_theta = delta_theta
        self.delta_constraint = delta_constraint or 0.1
        self.first_order = first_order
        self.mission_encoder = mission_encoder
        self.mission_adapter = mission_adapter
        self.constraint_adapter = constraint_adapter
        # Fixed per-hazard λ: {tile_index: weight}
        self.lambda_weights = lambda_weights or {2: 0.8, 3: 0.3, 4: 0.5}
     
    

    def adapt_one(self, mission):
        
        if mission is None:
            raise RuntimeError("Mission is None! Make sure each BatchEpisodes has a valid mission.")

        # Handle (goal, constraint) tuple or plain string
        if isinstance(mission, tuple):
            goal_text, constraint_text = mission
        else:
            goal_text = mission
            constraint_text = None

        # Goal path: encode goal text → MissionParamAdapter → Δθ_goal
        goal_emb = self.mission_encoder(goal_text)
        goal_emb = goal_emb.to(next(self.mission_adapter.parameters()).device)
        delta_goal = self.mission_adapter(goal_emb)
        delta_goal = [dg * self.delta_theta for dg in delta_goal]

        # Constraint path: encode constraint text → ConstraintParamAdapter → Δθ_constraint
        if constraint_text is not None and self.constraint_adapter is not None:
            constraint_emb = self.mission_encoder(constraint_text)
            constraint_emb = constraint_emb.to(next(self.constraint_adapter.parameters()).device)
            delta_constraint = self.constraint_adapter(constraint_emb)
            delta_constraint = [dc * self.delta_constraint for dc in delta_constraint]
        else:
            delta_constraint = None

        # θ' = θ + Δθ_goal + Δθ_constraint
        policy_params = list(self.policy.parameters())
        param_names = list(dict(self.policy.named_parameters()).keys())

        if delta_constraint is not None:
            theta_prime = OrderedDict(
                (name, param + dg.squeeze(0) + dc.squeeze(0))
                for name, param, dg, dc in zip(param_names, policy_params, delta_goal, delta_constraint)
            )
        else:
            theta_prime = OrderedDict(
                (name, param + dg.squeeze(0))
                for name, param, dg in zip(param_names, policy_params, delta_goal)
            )

        return theta_prime  
    

    def hessian_vector_product(self, kl, meta_params, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    meta_params,
                                    create_graph=True)      
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         meta_params,
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product


    def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        
        # Make sure train_futures and valid_futures are lists!
        if not isinstance(train_futures, list):
            train_futures = [train_futures]
        if not isinstance(valid_futures, list):
            valid_futures = [valid_futures]

        task_params_list = [self.adapt_one(getattr(valid_batch, "mission", None)) for valid_batch in valid_futures]

        with torch.set_grad_enabled(old_pi is None):
            losses = []
            kls = []
            old_pis = []
            for task_params, valid_episodes in zip(task_params_list, valid_futures):
                pi = self.policy(valid_episodes.observations, params=task_params)

                if old_pi is None:
                    old_pi_task = detach_distribution(pi)
                else:
                    old_pi_task = old_pi

                if isinstance(old_pi_task, list):
                    old_log_prob = old_pi_task[0].log_prob(valid_episodes.actions)
                else:
                    old_log_prob = old_pi_task.log_prob(valid_episodes.actions)
                log_ratio = pi.log_prob(valid_episodes.actions) - old_log_prob

                ratio = torch.exp(log_ratio)

                # Combined advantage: reward advantage - cost advantage (already weighted in sampler)
                if hasattr(valid_episodes, '_cost_advantages') and valid_episodes._cost_advantages is not None:
                    combined_adv = valid_episodes.advantages - valid_episodes.cost_advantages
                else:
                    combined_adv = valid_episodes.advantages

                loss = -weighted_mean(ratio * combined_adv,
                                    lengths=valid_episodes.lengths)
                
                if isinstance(old_pi_task, list):
                    kl_vals = [kl_divergence(pi, q) for q in old_pi_task]
                    kl = weighted_mean(torch.stack(kl_vals).mean(dim=0), lengths=valid_episodes.lengths)
                else:
                    kl = weighted_mean(kl_divergence(pi, old_pi_task), lengths=valid_episodes.lengths)

                losses.append(loss)
                kls.append(kl)
                old_pis.append(old_pi_task)

        return torch.stack(losses).mean(), torch.stack(kls).mean(), old_pis

    def step(self,
             train_futures,  
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures)
        logs = {}

        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(train_futures, valid_futures)])

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks

        meta_params = list(self.policy.parameters()) + list(self.mission_adapter.parameters()) + list(self.mission_encoder.parameters())
        if self.constraint_adapter is not None:
            meta_params += list(self.constraint_adapter.parameters())
        meta_params = [p for p in meta_params if p.requires_grad]
                        
        grads = torch.autograd.grad(old_loss,
                                    meta_params,
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,meta_params=meta_params,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(meta_params)

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 meta_params)

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(train_futures, valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks   
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params,meta_params)

        return logs
