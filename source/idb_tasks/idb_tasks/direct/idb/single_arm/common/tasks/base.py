"""Shared base class and configuration for all IDB Board #1 datacenter tasks."""

from __future__ import annotations

import torch

from idb_tasks.utils.isaac_compat import configclass

from ....wire_base.wire_state import ferrule_positions
from ..physics_metrics import (
    FigureEightPolishingTracker,
    NeighborDisturbanceMonitor,
    RetentionLatchState,
    SeatingResult,
    compute_cable_bend_radius,
    evaluate_connector_seating,
)
from ..scoring import RubricBreakdown


@configclass
class IDBTaskBaseCfg:
    """Common configuration inputs for every IDB Board #1 datacenter manipulation task."""

    task_class: type = object
    name: str = ""
    cable_preset: str = "os2_fiber_simplex"
    far_port: str = "B5"
    source_port: str = "A5"
    target_port: str = "A5"
    candidate_port_triples: list[list[str]] = []
    """List of [far_port, source_port, target_port] used when variation.randomize_target_port=True."""
    include_polishing: bool = False
    polishing_required_cycles: int = 10
    polishing_pressure_min_n: float = 1.0
    polishing_pressure_max_n: float = 2.0
    polishing_damage_force_n: float = 6.0
    roll_cos_threshold: float = 0.9659
    latch_release_force_n: float = 1.5
    latch_engage_hold_steps: int = 3
    damage_force_n: float = 35.0
    neighbor_disturbance_tol_m: float = 0.004

    def reserved_ports(self, include_candidates: bool = True) -> set[str]:
        ports = {self.far_port, self.source_port, self.target_port}
        if include_candidates:
            for triple in self.candidate_port_triples:
                ports.update(triple)
        return ports

    def observation_dim(self, joints: int) -> int:
        # 2*J + ee(7) + gripper(3) + wrench(6) + port(10) + source(6) + target(10) + status(7) = 2*J + 49
        base = 2 * joints + 49
        return base + (11 if self.include_polishing else 0)

    def resolve(self, env_cfg) -> None:
        from idb_assets.board_spec import load_board_spec

        board = load_board_spec(env_cfg.board.manifest_path)
        for pid in self.reserved_ports():
            board.port(pid)
        env_cfg.connector.family = board.port(self.source_port).family
        env_cfg.connector.far_family = board.port(self.far_port).family
        if board.port(self.target_port).family != env_cfg.connector.family:
            raise ValueError(
                f"Target port {self.target_port} family does not match source port {self.source_port}."
            )


class IDBDatacenterTaskBase:
    """Shared vectorized task state machine for IDB Board #1 cable-management tasks."""

    def __init__(self, env):
        self.env = env
        self.cfg: IDBTaskBaseCfg = env.cfg.task
        self.polishing: FigureEightPolishingTracker | None = None

    def initialize(self) -> None:
        env, cfg = self.env, self.cfg
        n, device = env.num_envs, env.device

        # Cache all same-family port indices on Board #1 so wrong-port insertions are caught.
        family = env.connector.family
        self._same_family_ids = [p.port_id for p in env.board.spec.ports_by_family(family)]
        self._same_family_indices = torch.tensor(
            [env.board.port_index[pid] for pid in self._same_family_ids], dtype=torch.long, device=device
        )

        self.far_port_idx = torch.full((n,), env.board.port_index[cfg.far_port], dtype=torch.long, device=device)
        self.source_port_idx = torch.full((n,), env.board.port_index[cfg.source_port], dtype=torch.long, device=device)
        self.target_port_idx = torch.full((n,), env.board.port_index[cfg.target_port], dtype=torch.long, device=device)

        self.removed_from_source = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever_kinked = torch.zeros(n, dtype=torch.bool, device=device)
        self.kinked_to_polish = torch.zeros(n, dtype=torch.bool, device=device)
        self.kinked_from_polish = torch.zeros(n, dtype=torch.bool, device=device)
        self.min_bend_radius_m = torch.full((n,), 1.0, device=device)

        self.latch = RetentionLatchState(
            num_envs=n,
            geometry=env.connector,
            device=device,
            latch_release_force_n=cfg.latch_release_force_n,
            engage_hold_steps=cfg.latch_engage_hold_steps,
        )
        self.neighbors = NeighborDisturbanceMonitor(
            num_envs=n,
            num_neighbors=len(env.cfg.resolved_neighbor_cables),
            disturbance_pos_tol_m=cfg.neighbor_disturbance_tol_m,
            damage_force_n=cfg.damage_force_n,
            device=device,
        )
        if cfg.include_polishing:
            self.polishing = FigureEightPolishingTracker(
                num_envs=n,
                required_cycles=cfg.polishing_required_cycles,
                pressure_min_n=cfg.polishing_pressure_min_n,
                pressure_max_n=cfg.polishing_pressure_max_n,
                damage_force_n=cfg.polishing_damage_force_n,
                device=device,
            )

        self.source_pos_w = torch.zeros(n, 3, device=device)
        self.source_quat_w = torch.zeros(n, 4, device=device)
        self.target_pos_w = torch.zeros(n, 3, device=device)
        self.target_quat_w = torch.zeros(n, 4, device=device)
        self.source_seating: SeatingResult | None = None
        self.target_seating: SeatingResult | None = None
        self.seated_any_port = torch.zeros(n, dtype=torch.bool, device=device)
        self.seated_port_idx = torch.full((n,), -1, dtype=torch.long, device=device)
        self.inserted_correct_port = torch.zeros(n, dtype=torch.bool, device=device)
        self.wrong_port_inserted = torch.zeros(n, dtype=torch.bool, device=device)

    def _gather_port_poses(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_pos, all_quat = self.env.board.all_ports_world_pose()
        batch = torch.arange(self.env.num_envs, device=self.env.device)
        return all_pos[batch, indices], all_quat[batch, indices]

    def reset(
        self, env_ids: torch.Tensor, nominal_neighbors: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset task trackers and return world (far_pos, far_quat, source_pos, source_quat) for cable reset."""
        env, cfg = self.env, self.cfg
        count = len(env_ids)
        device = env.device

        if env.cfg.variation.randomize_target_port and cfg.candidate_port_triples:
            choices = torch.randint(0, len(cfg.candidate_port_triples), (count,), device=device)
            triples = cfg.candidate_port_triples
            far_ids = [env.board.port_index[triples[int(c)][0]] for c in choices]
            src_ids = [env.board.port_index[triples[int(c)][1]] for c in choices]
            tgt_ids = [env.board.port_index[triples[int(c)][2]] for c in choices]
            self.far_port_idx[env_ids] = torch.tensor(far_ids, dtype=torch.long, device=device)
            self.source_port_idx[env_ids] = torch.tensor(src_ids, dtype=torch.long, device=device)
            self.target_port_idx[env_ids] = torch.tensor(tgt_ids, dtype=torch.long, device=device)
        else:
            self.far_port_idx[env_ids] = env.board.port_index[cfg.far_port]
            self.source_port_idx[env_ids] = env.board.port_index[cfg.source_port]
            self.target_port_idx[env_ids] = env.board.port_index[cfg.target_port]

        self.removed_from_source[env_ids] = False
        self.ever_kinked[env_ids] = False
        self.kinked_to_polish[env_ids] = False
        self.kinked_from_polish[env_ids] = False
        self.min_bend_radius_m[env_ids] = 1.0
        self.inserted_correct_port[env_ids] = False
        self.wrong_port_inserted[env_ids] = False

        self.latch.reset(env_ids, initial_port_idx=self.source_port_idx[env_ids], initially_latched=True)
        self.neighbors.reset(env_ids, nominal_positions=nominal_neighbors)
        if self.polishing is not None:
            self.polishing.reset(env_ids)

        # Compute seated poses for far_port and source_port of each reset env.
        far_pos_list, far_quat_list, src_pos_list, src_quat_list = [], [], [], []
        for eid in env_ids.tolist():
            f_pid = env.board.port_ids[int(self.far_port_idx[eid])]
            s_pid = env.board.port_ids[int(self.source_port_idx[eid])]
            fp, fq = env.board.seated_connector_pose(f_pid, env.far_connector)
            sp, sq = env.board.seated_connector_pose(s_pid, env.connector)
            far_pos_list.append(fp[eid])
            far_quat_list.append(fq[eid])
            src_pos_list.append(sp[eid])
            src_quat_list.append(sq[eid])

        return (
            torch.stack(far_pos_list, dim=0),
            torch.stack(far_quat_list, dim=0),
            torch.stack(src_pos_list, dim=0),
            torch.stack(src_quat_list, dim=0),
        )

    def refresh_after_reset(self, env_ids: torch.Tensor) -> None:
        """Refresh seating geometry and initial bend metric after reset without advancing latch or fault state."""
        env, cfg = self.env, self.cfg
        state = env.wire_state
        if state is None:
            return
        self.source_pos_w, self.source_quat_w = self._gather_port_poses(self.source_port_idx)
        self.target_pos_w, self.target_quat_w = self._gather_port_poses(self.target_port_idx)
        self.source_seating = evaluate_connector_seating(
            state.insertion_end_pos,
            state.insertion_end_quat,
            self.source_pos_w,
            self.source_quat_w,
            env.connector,
            roll_cos_threshold=cfg.roll_cos_threshold,
        )
        self.target_seating = evaluate_connector_seating(
            state.insertion_end_pos,
            state.insertion_end_quat,
            self.target_pos_w,
            self.target_quat_w,
            env.connector,
            roll_cos_threshold=cfg.roll_cos_threshold,
        )
        self.seated_any_port[env_ids] = True
        self.seated_port_idx[env_ids] = self.source_port_idx[env_ids]
        bend = compute_cable_bend_radius(state.centerline, min_bend_radius=env.cfg.wire.min_bend_radius)
        self.min_bend_radius_m[env_ids] = bend.min_radius_m[env_ids]

    def update(self) -> None:
        env, cfg = self.env, self.cfg
        state = env.wire_state
        if state is None:
            return

        self.source_pos_w, self.source_quat_w = self._gather_port_poses(self.source_port_idx)
        self.target_pos_w, self.target_quat_w = self._gather_port_poses(self.target_port_idx)

        self.source_seating = evaluate_connector_seating(
            state.insertion_end_pos,
            state.insertion_end_quat,
            self.source_pos_w,
            self.source_quat_w,
            env.connector,
            roll_cos_threshold=cfg.roll_cos_threshold,
        )
        self.target_seating = evaluate_connector_seating(
            state.insertion_end_pos,
            state.insertion_end_quat,
            self.target_pos_w,
            self.target_quat_w,
            env.connector,
            roll_cos_threshold=cfg.roll_cos_threshold,
        )

        # Check seating across ALL same-family ports on the board.
        all_pos, all_quat = env.board.all_ports_world_pose()
        seated_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        seated_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        aligned_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for p_idx in self._same_family_indices.tolist():
            res = evaluate_connector_seating(
                state.insertion_end_pos,
                state.insertion_end_quat,
                all_pos[:, p_idx],
                all_quat[:, p_idx],
                env.connector,
                roll_cos_threshold=cfg.roll_cos_threshold,
            )
            hit = res.is_seated & (~seated_any)
            seated_any |= res.is_seated
            seated_idx = torch.where(hit, torch.full_like(seated_idx, p_idx), seated_idx)
            aligned_any |= res.is_seated & res.key_aligned
        self.seated_any_port = seated_any
        self.seated_port_idx = seated_idx

        # Removal from source: connector leading face is completely clear of source mouth (tip_depth < -0.005 m).
        clear_of_source = (~self.source_seating.is_seated) & (self.source_seating.tip_depth < -0.005)
        self.removed_from_source |= clear_of_source

        # Retention latch state update.
        axial_pull = env.arm.wrench[:, :3].norm(dim=-1)
        self.latch.update(
            is_seated_on_any=seated_any,
            seated_port_idx=seated_idx,
            key_aligned=aligned_any,
            is_grasped=env.arm.is_grasping,
            gripper_latch_force_n=env.arm.gripper_pinch_force,
            axial_pull_force_n=axial_pull,
        )

        # Cable bend radius / kink check.
        bend = compute_cable_bend_radius(state.centerline, min_bend_radius=env.cfg.wire.min_bend_radius)
        self.min_bend_radius_m = bend.min_radius_m
        self.ever_kinked |= bend.is_kinked

        # Polishing tracker for fiber maintenance tasks (Tasks 4 & 5).
        if self.polishing is not None:
            pad_c, pad_n, pad_r = env.board.polishing_pad_world_pose()
            ferrules = ferrule_positions(state.insertion_end_pos, state.insertion_end_quat, env.connector)
            pad_force = env.board.polishing_contact_force_n()
            self.polishing.update(ferrules, pad_c, pad_n, pad_r, pad_force)
            before_clean = self.removed_from_source & (~self.polishing.cycles_ok)
            after_clean = self.polishing.cycles_ok
            self.kinked_to_polish |= before_clean & bend.is_kinked
            self.kinked_from_polish |= after_clean & bend.is_kinked

        # Neighbor integrity & excessive force monitor.
        max_force = torch.maximum(env.arm.wrench[:, :3].norm(dim=-1), env.board.polishing_contact_force_n())
        self.neighbors.update(
            current_positions=env.board.neighbor_active_positions(),
            max_active_contact_force_n=max_force,
        )

        # Track whether active connector has been properly seated into the target vs wrong port after removal.
        at_target = self.removed_from_source & self.target_seating.is_seated
        if self.polishing is not None:
            at_target = at_target & self.polishing.cycles_ok
        self.inserted_correct_port |= at_target
        self.wrong_port_inserted = self.removed_from_source & seated_any & (seated_idx != self.target_port_idx)

    def damaged(self) -> torch.Tensor:
        dmg = self.latch.damaged | (~self.neighbors.integrity_ok)
        if self.polishing is not None:
            dmg = dmg | self.polishing.ferrule_damaged
        return dmg

    def evaluate(self) -> RubricBreakdown:
        raise NotImplementedError

    def success(self) -> torch.Tensor:
        return self.evaluate().passed

    def observations(self) -> torch.Tensor:
        env, arm = self.env, self.env.arm
        origins = env.scene.env_origins
        src_local = self.source_pos_w - origins
        tgt_local = self.target_pos_w - origins
        port = env.port_local_pos
        tip = env.tip_local_pos
        wrench = arm.wrench if env.cfg.sensors.enable_wrist_ft else torch.zeros_like(arm.wrench)

        tip_depth = self.target_seating.tip_depth if self.target_seating is not None else torch.zeros(env.num_envs, device=env.device)
        roll_cos = self.target_seating.roll_cos if self.target_seating is not None else torch.ones(env.num_envs, device=env.device)

        parts = [
            arm.joint_pos,
            arm.joint_vel,
            env.ee_local_pos,
            env.ee_local_quat,
            arm.opening.unsqueeze(-1),
            torch.clamp(arm.force_left.unsqueeze(-1) / 10.0, 0.0, 1.0),
            torch.clamp(arm.force_right.unsqueeze(-1) / 10.0, 0.0, 1.0),
            wrench,
            port,
            env.wire_state.insertion_end_quat,
            tip,
            src_local,
            src_local - port,
            tgt_local,
            tgt_local - port,
            self.target_quat_w,
            self.removed_from_source.float().unsqueeze(-1),
            self.latch.engaged.float().unsqueeze(-1),
            tip_depth.unsqueeze(-1),
            roll_cos.unsqueeze(-1),
            self.min_bend_radius_m.unsqueeze(-1),
            self.ever_kinked.float().unsqueeze(-1),
            (~self.damaged()).float().unsqueeze(-1),
        ]
        if self.cfg.include_polishing and self.polishing is not None:
            pad_c, _, _ = env.board.polishing_pad_world_pose()
            pad_local = pad_c - origins
            ferrules = ferrule_positions(env.wire_state.insertion_end_pos, env.wire_state.insertion_end_quat, env.connector)
            ferrule_mean_local = ferrules.mean(dim=1) - origins
            pad_force = env.board.polishing_contact_force_n()
            cycle_frac = (self.polishing.cycles_completed.float() / max(self.cfg.polishing_required_cycles, 1)).clamp(0.0, 1.0)
            parts.extend([
                pad_local,
                pad_local - ferrule_mean_local,
                pad_force.unsqueeze(-1),
                self.polishing.in_contact.float().unsqueeze(-1),
                cycle_frac.unsqueeze(-1),
                self.polishing.continuous_contact_ok.float().unsqueeze(-1),
                self.polishing.pressure_ok.float().unsqueeze(-1),
            ])
        return torch.cat(parts, dim=-1)
