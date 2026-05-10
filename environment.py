from minigrid.envs.babyai.goto import GoToLocal, GoToObjDoor, GoTo
from minigrid.envs.babyai.open import Open
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc, OpenInstr, PickupInstr, BeforeInstr, AfterInstr
from minigrid.core.world_object import (Ball, Box, Key, Lava, WorldObj,
                                         OBJECT_TO_IDX, IDX_TO_OBJECT,
                                         fill_coords, point_in_rect, point_in_line)
import re
import numpy as np
import itertools


OBJECT_TO_IDX['grass'] = 11
OBJECT_TO_IDX['water'] = 12
IDX_TO_OBJECT[11] = 'grass'
IDX_TO_OBJECT[12] = 'water'


class Grass(WorldObj):
    def __init__(self):
        super().__init__('grass', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (172, 229, 102)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Water(WorldObj):
    def __init__(self):
        super().__init__('water', 'blue')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (31, 206, 203)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class SafeLava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        fill_coords(img, point_in_rect(0.2, 0.8, 0.3, 0.7), (255, 50, 0))


# ──────────────────────────────────────────────────────────────────────
# Tile index scheme: map every cell type to a simple integer
# ──────────────────────────────────────────────────────────────────────

TILE_INDEX = {
    None:    0,  # empty floor
    'wall':  0,  # impassable — agent never stands here
    'floor': 0,  # explicit floor tile
    'door':  0,  # door treated as floor
    'goal':  0,  # goal tile
    'ball':  1,  # object
    'key':   1,  # object
    'box':   1,  # object
    'lava':  2,  # hazard
    'grass': 3,  # hazard
    'water': 4,  # hazard
}

HAZARD_TYPES = {'lava': 2, 'grass': 3, 'water': 4}
HAZARD_CLASSES = {'lava': SafeLava, 'grass': Grass, 'water': Water}

def get_tile_index(cell):
    """Return the tile index (0-4) for a grid cell."""
    if cell is None:
        return 0
    return TILE_INDEX.get(cell.type, 0)


OBJECTS = ['ball', 'key']
COLORS = ['red', 'green', 'blue', 'purple']
PREP_LOCS = ['on', 'at', 'to']

# Location names
LOC_NAMES = ['left', 'behind']

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]

# Constraints
CONSTRAINT_TEXTS = [f"avoid {hazard}" for hazard in HAZARD_TYPES]
DOUBLE_CONSTRAINT_TEXTS = [f"avoid {h1} and avoid {h2}" for h1, h2 in itertools.combinations(HAZARD_TYPES.keys(), 2)]

# Combined missions (used by env's set_forced_mission)
CONSTRAINED_LOCAL_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in LOCAL_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_LOCAL_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in LOCAL_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

CONSTRAINED_PICKUP_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in PICKUP_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_PICKUP_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in PICKUP_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For GoToObjDoor, GoToOpen environments  
DOOR_MISSIONS = [f"go to the {color} door" for color in COLORS]

GOTOOBJDOOR_MISSIONS = LOCAL_MISSIONS + DOOR_MISSIONS

CONSTRAINED_GOTOOBJDOOR_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in GOTOOBJDOOR_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

CONSTRAINED_GOTOOPEN_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in LOCAL_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_GOTOOPEN_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in LOCAL_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For OpenDoor Environment
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in COLORS]

CONSTRAINED_OPENDOOR_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOOR_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_OPENDOOR_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOOR_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For OpenDoorLoc Environment
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]

CONSTRAINED_OPENDOORLOC_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_OPENDOORLOC_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For OpenDoorsOrder
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in COLORS for c2 in COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in COLORS for c2 in COLORS]
)

CONSTRAINED_OPENDOORSORDER_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOORS_ORDER_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_OPENDOORSORDER_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in OPEN_DOORS_ORDER_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For ActionObjDoor Environment
ACTIONOBJDOOR_MISSIONS = (
    [f"pick up the {c} {t}" for c in COLORS for t in ["ball", "key"]] +
    [f"go to the {c} {t}" for c in COLORS for t in ["ball", "key", "door"]] +
    [f"open the {c} door" for c in COLORS]
)

CONSTRAINED_ACTIONOBJDOOR_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in ACTIONOBJDOOR_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_ACTIONOBJDOOR_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in ACTIONOBJDOOR_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# For FindObjS5 Environment
FINDOBJS5_MISSIONS = [f"pick up the {t}" for t in ["ball", "key"]]

CONSTRAINED_FINDOBJS5_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in FINDOBJS5_MISSIONS for constraint in CONSTRAINT_TEXTS
]

DOUBLE_CONSTRAINED_FINDOBJS5_MISSIONS = [
    f"{goal} and {constraint}"
    for goal in FINDOBJS5_MISSIONS for constraint in DOUBLE_CONSTRAINT_TEXTS
]

# ---------------------------------------------------------------------


class GoToLocalMissionEnv(GoToLocal):
    def __init__(self, room_size=6, num_dists=2, **kwargs):
        super().__init__(room_size=room_size, num_dists=num_dists, **kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):
        
        if self._forced_mission is not None:
            m = re.match(r"go to the (\w+) (\w+)", self._forced_mission) 
            if m:
                color, obj_type = m.groups()
                self.agent_pos = np.array((1,1))
                
                # Map string to class
                OBJ_CLASS = {"ball": Ball, "box": Box, "key": Key}
                target_obj = OBJ_CLASS[obj_type](color)

                i = self._rand_int(0, self.num_cols)
                j = self._rand_int(0, self.num_rows)
                target_obj, _ = self.add_object(i, j, obj_type, color)

                # Add distractors 
                distractor_count = 0
                choices = [(t, c) for t in OBJECTS for c in COLORS if not (t == obj_type and c == color)]
                while distractor_count < self.num_dists:
                    dist_type, dist_color = self._rand_elem(choices)
                    try:
                        i = self._rand_int(0, self.num_cols)
                        j = self._rand_int(0, self.num_rows)
                        self.add_object(i, j, dist_type, dist_color)
                        distractor_count += 1
                    except Exception:
                        continue
                self.instrs = GoToInstr(ObjDesc(obj_type, color))
                self.check_objs_reachable()
            else:
                super().gen_mission()
        else:
            super().gen_mission()




class ConstrainedGoToLocalEnv(GoToLocalMissionEnv):
    """GoToLocal + random hazard tiles + per-step cost signal.

    The mission is e.g. "go to the blue ball and avoid lava".
    On every step, info['tile_index'] gives the tile type under the agent
    and info['cost'] is 1.0 if that tile matches the constraint hazard.
    """

    def __init__(self, room_size=8, num_dists=2, hazard_density=0.15, max_hazards=4, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []  # set by set_forced_mission
        super().__init__(room_size=room_size, num_dists=num_dists, **kwargs)

    def set_forced_mission(self, mission):
        # Parse "go to the blue ball and avoid lava" or "avoid lava and avoid water"
        super().set_forced_mission(mission)
        self.constraint_tiles = []
        for hazard, idx in HAZARD_TYPES.items():
            if f"avoid {hazard}" in mission:
                self.constraint_tiles.append(idx)

    def gen_mission(self):
        super().gen_mission()  # places goal + distractors
        
        # Place hazard classes that correspond to the constraints
        for constraint_tile in self.constraint_tiles:
            hazard_cls = None
            for name, idx in HAZARD_TYPES.items():
                if idx == constraint_tile:
                    hazard_cls = HAZARD_CLASSES[name]
                    break
            
            if hazard_cls is not None:
                # Find all valid empty cells
                empty_cells = []
                for i in range(self.grid.width):
                    for j in range(self.grid.height):
                        if (self.grid.get(i, j) is None
                                and not np.array_equal(self.agent_pos, (i, j))):
                            empty_cells.append((i, j))
                
                # Calculate how many to place (based on density) but cap it at 4
                num_to_place = 0
                for _ in range(len(empty_cells)):
                    if self._rand_float(0, 1) < self.hazard_density:
                        num_to_place += 1
                
                num_to_place = min(num_to_place, self.max_hazards)
                
                # If we have cells and need to place some, randomly select them
                if num_to_place > 0 and empty_cells:
                    self.np_random.shuffle(empty_cells)
                    for pos in empty_cells[:num_to_place]:
                        self.grid.set(pos[0], pos[1], hazard_cls())

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        tile_idx = get_tile_index(cell)
        info['tile_index'] = tile_idx
        info['cost'] = 1.0 if tile_idx in self.constraint_tiles else 0.0
        # MiniGrid base kills the agent on 'lava' type cells — undo that.
        # In our constrained env, hazards only incur cost, never terminate.
        if terminated and cell is not None and cell.type in ('lava', 'grass', 'water'):
            terminated = False
            reward = 0  # don't give the negative lava reward either
        return obs, reward, terminated, truncated, info


class PickupDistMissionEnv(RoomGridLevel):
    def __init__(self, debug=False, room_size=7, num_dists=5,
                 num_rows=1, num_cols=1, max_steps=200, **kwargs):
        self.debug = debug
        super().__init__(num_rows=num_rows, num_cols=num_cols,
                         max_steps=max_steps, room_size=room_size, **kwargs)
        self.num_dists = num_dists
        self.fixed_max_steps = True
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def gen_mission(self):
        if self._forced_mission is not None:
            m = re.match(r"pick up (a|the) (?:(\w+)\s)?(\w+)?", self._forced_mission)
            if m:
                _, color, obj_type = m.groups()
                color    = color if color not in (None, "box", "ball", "key") else None
                obj_type = obj_type if obj_type in ("box", "ball", "key") else None

                self.place_agent(0, 0)

                placed = False
                if obj_type is not None and color is not None:
                    attempts = 0
                    while not placed and attempts < 50:
                        try:
                            x = self._rand_int(0, self.num_cols)
                            y = self._rand_int(0, self.num_rows)
                            self.add_object(x, y, obj_type, color)
                            placed = True
                        except Exception:
                            attempts += 1

                distractors_to_add = max(self.num_dists, 0)
                choices = [(t, c) for t in OBJECTS for c in COLORS]
                if obj_type is not None and color is not None:
                    choices = [(t, c) for (t, c) in choices
                               if not (t == obj_type and c == color)]
                distractor_count = 0
                while distractor_count < distractors_to_add:
                    dist_type, dist_color = self._rand_elem(choices)
                    try:
                        x = self._rand_int(0, self.num_cols)
                        y = self._rand_int(0, self.num_rows)
                        self.add_object(x, y, dist_type, dist_color)
                        distractor_count += 1
                    except Exception:
                        continue

                if not placed:
                    t_choice = obj_type if obj_type is not None else self._rand_elem(OBJECTS)
                    c_choice = color    if color    is not None else self._rand_elem(COLORS)
                    try:
                        x = self._rand_int(0, self.num_cols)
                        y = self._rand_int(0, self.num_rows)
                        self.add_object(x, y, t_choice, c_choice)
                    except Exception:
                        pass

                self.instrs = PickupInstr(ObjDesc(obj_type, color), strict=self.debug)
                self.mission = self.instrs.surface(self)
                return

        objs = self.add_distractors(num_distractors=self.num_dists)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type_, color_ = obj.type, obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type_ = None
        elif select_by == "type":
            color_ = None

        self.instrs = PickupInstr(ObjDesc(type_, color_), strict=self.debug)
        self.mission = self.instrs.surface(self)
        return

    def _matches_desc(self, obj, desc):
        if obj is None:
            return False
        if (desc.type  is not None) and (obj.type  != desc.type):
            return False
        if (desc.color is not None) and (obj.color != desc.color):
            return False
        return True

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        carrying = getattr(self, "carrying", None)
        info["carrying"] = carrying

        if not terminated and isinstance(self.instrs, PickupInstr):
            if self._matches_desc(carrying, self.instrs.desc):
                terminated = True
                info["success"] = True
                if reward == 0:
                    try:
                        reward = self._reward()
                    except Exception:
                        pass

        return obs, reward, terminated, truncated, info



class GoToObjDoorMissionEnv(GoToObjDoor):
    def __init__(self,num_distractors=3, **kwargs):
        super().__init__( **kwargs)
        self._forced_mission = None
        self.num_distractors = num_distractors
        self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):

        if self._forced_mission is not None:
            m = re.match(r"go to (a|the) (\w+) (\w+)", self._forced_mission)
            if m:
                _, color, obj_type = m.groups()
                self.place_agent(1, 1)
                objs = []
                if obj_type == "door":
                    # Pick up to 4 door colors, one is the target, rest are distractors
                    distractor_colors = [c for c in COLORS if c != color]
                    all_door_colors = [color] + distractor_colors
                    all_door_colors = all_door_colors[:4]

                    # Shuffle target to change position
                    self.np_random.shuffle(all_door_colors)
                    for dcolor in all_door_colors:
                        door, _ = self.add_door(1, 1, color=dcolor)
                        objs.append(door)

                else:
                    obj, _ = self.add_object(1, 1, obj_type, color)
                    objs.append(obj)

                    # Add distractors
                    distractor_count = 0
                    while distractor_count < self.num_distractors:
                        distractor_color = self._rand_elem([c for c in COLORS if c != color])
                        distractor_type = self._rand_elem([t for t in OBJECTS if t != obj_type])
                        try:
                            self.add_object(1, 1, distractor_type, distractor_color)
                            distractor_count += 1
                        except Exception:
                            continue

                self.check_objs_reachable()
                self.instrs = GoToInstr(ObjDesc(obj_type, color))
            else:
                super().gen_mission()
        else:
            super().gen_mission()


# -----------------------------------------------------------------------

class GoToOpenMissionEnv(GoTo):
    def __init__(self, room_size=6, num_rows=2, num_cols=2, num_dists=10, **kwargs):
        super().__init__(room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         num_dists=num_dists,
                         doors_open=True,
                         **kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def gen_mission(self):
        if self._forced_mission is not None:
            m = re.match(r"go to (a|the) (\w+) (\w+)", self._forced_mission)
            if m:
                _, color, obj_type = m.groups()

                self.place_agent()
                self.connect_all()

                if obj_type == "door":
                # verify at least one door color of forced mission exists
                    doors = []
                    for i in range(self.num_cols):
                        for j in range(self.num_rows):
                            room = self.get_room(i, j)
                            for door in room.doors:
                                if door and door.color == color:
                                    doors.append(door)
                    if not doors:
                        door, _ = self.add_door(0, 0, color=color, locked=False)
                    self.instrs = GoToInstr(ObjDesc("door", color))

                else:
                    # Add target object
                    obj, _ = self.add_object(self._rand_int(0, self.num_cols),
                                            self._rand_int(0, self.num_rows),
                                            obj_type, color)

                    # Add distractors from all except the exact target
                    choices = [(t, c) for t in OBJECTS for c in COLORS
                            if not (t == obj_type and c == color)]
                    placed = 0
                    while placed < self.num_dists:
                        dist_type, dist_color = self._rand_elem(choices)
                        try:
                            self.add_object(self._rand_int(0, self.num_cols),
                                            self._rand_int(0, self.num_rows),
                                            dist_type, dist_color)
                            placed += 1
                        except Exception:
                            continue


                    self.check_objs_reachable()
                    self.instrs = GoToInstr(ObjDesc(obj_type, color))
                    self.open_all_doors()
                    return
        super().gen_mission()


class OpenDoorMissionEnv(Open):
    def __init__(self, debug = False, **kwargs):
        self.debug = debug
        super().__init__(**kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def _valid_sides(self, i: int, j: int):
        # 0=E, 1=S, 2=W, 3=N; only sides that have a neighboring room
        sides = []
        if i + 1 < self.num_cols: sides.append(0)
        if j + 1 < self.num_rows: sides.append(1)
        if i - 1 >= 0:            sides.append(2)
        if j - 1 >= 0:            sides.append(3)
        return sides

    def gen_mission(self):
        i = j = 1
        self.place_agent(i, j)

        color = None
        if self._forced_mission:
            m = re.match(r"\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
            if m:
                color = m.group(1).lower()
        if color is None:
            color = self._rand_elem(COLORS)

        # Choose forced color 
        target_color = color
        sides = self._valid_sides(i, j)
        self.np_random.shuffle(sides)

        # Place the target door first
        target_side = sides[0]
        target_door, _ = self.add_door(i, j, door_idx=target_side, color=target_color, locked=False)

        # Place distractor doors on remaining sides with different colors
        distractor_colors = [c for c in COLORS if c != target_color]
        distractor_colors = self._rand_subset(distractor_colors, len(sides) - 1)
        for side, dcol in zip(sides[1:], distractor_colors):
            self.add_door(i, j, door_idx=side, color=dcol, locked=False)

        self.instrs = OpenInstr(ObjDesc(target_door.type, target_color), strict=self.debug)
        self.mission = self.instrs.surface(self)


class OpenDoorLocMissionEnv(Open):
    def __init__(self, debug=False, select_by=None, **kwargs):
        self.debug = debug
        self.select_by = select_by  
        super().__init__(**kwargs)
        self._forced_mission = None
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def _valid_sides(self, i: int, j: int):
        # 0=E, 1=S, 2=W, 3=N (only sides that have a neighboring room)
        sides = []
        if i + 1 < self.num_cols: sides.append(0)
        if j + 1 < self.num_rows: sides.append(1)
        if i - 1 >= 0:            sides.append(2)
        if j - 1 >= 0:            sides.append(3)
        return sides

    def gen_mission(self):
        i = j = 1
        self.place_agent(i, j)

        # Selection mode (color vs loc)
        select_by = self.select_by
        forced_color = None
        forced_loc = None
        if self._forced_mission:
            m_color = re.match(r"\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
            m_loc   = re.match(r"\s*open\s+(?:a|the)\s+door\s+(?:on|at|to)\s+the\s+(\w+)\s*$", self._forced_mission, re.IGNORECASE)
            if m_color:
                forced_color = m_color.group(1).lower()
                select_by = "color"
            elif m_loc:
                forced_loc = m_loc.group(1).lower()

                select_by = "loc"

        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])

        # Add Doors
        sides = self._valid_sides(i, j)
        self.np_random.shuffle(sides)
        first_door = None 

        if select_by == "color":
            # target forced color
            target_color = forced_color if forced_color is not None else self._rand_elem(COLORS)

            # place target door
            target_side = sides[0]
            target_door, _ = self.add_door(i, j, door_idx=target_side, color=target_color, locked=False)
            first_door = target_door if first_door is None else first_door

            # distractor doors 
            distractor_colors = [c for c in COLORS if c != target_color]
            distractor_colors = self._rand_subset(distractor_colors, max(0, len(sides) - 1))
            for side, dcol in zip(sides[1:], distractor_colors):
                d, _ = self.add_door(i, j, door_idx=side, color=dcol, locked=False)
                if first_door is None: first_door = d

            self.instrs = OpenInstr(ObjDesc(target_door.type, color=target_color), strict=self.debug)

        else:  
            # place doors 
            door_colors = self._rand_subset(COLORS, len(sides))
            placed = []
            for side, col in zip(sides, door_colors):
                d, _ = self.add_door(i, j, door_idx=side, color=col, locked=False)
                placed.append(d)
            if placed:
                first_door = placed[0]

            # pick location: forced or random from LOC_NAMES
            loc = forced_loc if forced_loc in LOC_NAMES else None
            if loc is None:
                loc = forced_loc if forced_loc in LOC_NAMES else self._rand_elem(LOC_NAMES)

            self.instrs = OpenInstr(ObjDesc(first_door.type, loc=loc), strict=self.debug)

        self.mission = self.instrs.surface(self)


class OpenDoorsOrderMissionEnv(Open):
    def __init__(self, num_doors=2, debug=False, room_size=None, max_steps=None, **kwargs):
        assert num_doors >= 1
        self.num_doors = num_doors
        self.debug = debug
        self._forced_mission = None
        if room_size is None:
            room_size = 6
        if max_steps is None:
            max_steps = 20 * (room_size ** 2)
        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        self._forced_mission = mission

    def _parse_forced(self):
        if not self._forced_mission:
            return None
        m1 = re.match(r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s*$", self._forced_mission, re.IGNORECASE)
        if m1:
            return ("single", m1.group(1).lower(), None)
        m2 = re.match(
            r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s*,?\s*(?:then|and\s+then)\s+open\s+(?:a|the)\s+(\w+)\s+door\s*$",
            self._forced_mission, re.IGNORECASE
        )
        if m2:
            return ("before", m2.group(1).lower(), m2.group(2).lower())
        m3 = re.match(
            r"^\s*open\s+(?:a|the)\s+(\w+)\s+door\s+after\s+you\s+open\s+(?:a|the)\s+(\w+)\s+door\s*$",
            self._forced_mission, re.IGNORECASE
        )
        if m3:
            return ("after", m3.group(1).lower(), m3.group(2).lower())
        return None

    def gen_mission(self):
        spec = self._parse_forced()
        if spec is None:
            colors = list(self._rand_subset(COLORS, min(self.num_doors, len(COLORS))))
        else:
            mode, c1, c2 = spec
            need = [c1] if mode == "single" else [c1, c2]
            pool = list(COLORS)
            self.np_random.shuffle(pool)
            extras = [c for c in pool if c not in need]
            take = max(0, self.num_doors - len(need))
            colors = need + extras[:take]

        i = j = 1
        self.place_agent(i, j)

        doors = []
        for k in range(self.num_doors):
            d, _ = self.add_door(i, j, color=colors[k], locked=False)
            doors.append(d)

        if spec is None:
            if self.num_doors == 1:
                desc1 = ObjDesc(doors[0].type, doors[0].color)
                self.instrs = OpenInstr(desc1, strict=self.debug)
            else:
                d1, d2 = self._rand_subset(doors, 2)
                desc1 = ObjDesc(d1.type, d1.color)
                desc2 = ObjDesc(d2.type, d2.color)
                if self._rand_int(0, 2) == 0:
                    self.instrs = BeforeInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
                else:
                    self.instrs = AfterInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        else:
            mode, c1, c2 = spec
            if mode == "single":
                d1 = next((d for d in doors if d.color == c1), doors[0])
                self.instrs = OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug)
            elif mode == "before":
                d1 = next((d for d in doors if d.color == c1), doors[0])
                d2 = next((d for d in doors if d.color == c2), doors[-1 if len(doors) > 1 else 0])
                self.instrs = BeforeInstr(
                    OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug),
                    OpenInstr(ObjDesc(d2.type, d2.color), strict=self.debug),
                )
            else:
                d1 = next((d for d in doors if d.color == c1), doors[0])
                d2 = next((d for d in doors if d.color == c2), doors[-1 if len(doors) > 1 else 0])
                self.instrs = AfterInstr(
                    OpenInstr(ObjDesc(d1.type, d1.color), strict=self.debug),
                    OpenInstr(ObjDesc(d2.type, d2.color), strict=self.debug),
                )

        self.mission = self.instrs.surface(self)
        self.check_objs_reachable()


# ----------------------------------------------------------------------

class ActionObjDoorMissionEnv(RoomGridLevel):
    def __init__(self, room_size=8, num_dists=2, max_steps=200, **kwargs):
        self.num_dists = num_dists
        self._forced_mission = None
        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=self.num_dists)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)
        
        target_obj = None
        if self._forced_mission is not None:
            m = re.match(r"(pick up|go to|open) (?:a|an|the) (?:(\w+)\s)?(\w+)?", self._forced_mission)
            if m:
                action_str, color, obj_type = m.groups()
                # Handle cases where color is omitted (e.g. "go to the door")
                if color in ("ball", "box", "key", "door"):
                    obj_type = color
                    color = None
                
                # Try to find a matching object in the distractors/doors we already placed
                for o in objs:
                    if o.type == obj_type and (color is None or o.color == color):
                        target_obj = o
                        break
                
                # If we didn't randomly generate the required target, forcefully add it
                if target_obj is None:
                    if obj_type == "door":
                        # We already added 4 doors, walls might be full. Recoloring an existing door.
                        door_objs = [o for o in objs if o.type == "door"]
                        if door_objs:
                            target_obj = self._rand_elem(door_objs)
                            if color is not None:
                                target_obj.color = color
                        else:
                            raise ValueError(f"No doors available to recolor for forced mission: {self._forced_mission}")
                    else:
                        try:
                            target_obj, _ = self.add_object(1, 1, kind=obj_type, color=color)
                            objs.append(target_obj)
                        except Exception:
                            # Fallback if add_object fails (e.g., room full)
                            same_type_objs = [o for o in objs if o.type == obj_type]
                            if same_type_objs:
                                target_obj = self._rand_elem(same_type_objs)
                                if color is not None:
                                    target_obj.color = color
                            else:
                                raise ValueError(f"Failed to spawn {color} {obj_type} and no existing ones to recolor.")
                
                desc = ObjDesc(target_obj.type, target_obj.color)
                if action_str == "go to":
                    self.instrs = GoToInstr(desc)
                elif action_str == "pick up":
                    self.instrs = PickupInstr(desc)
                elif action_str == "open":
                    self.instrs = OpenInstr(desc)
            else:
                raise ValueError(f"Failed to parse forced mission: {self._forced_mission}")
        else:
            # Fallback only for initial environment creation
            target_obj = self._rand_elem(objs)
            
        if target_obj is not None and getattr(self, "instrs", None) is None:
            desc = ObjDesc(target_obj.type, target_obj.color)
            if target_obj.type == "door":
                if self._rand_bool():
                    self.instrs = GoToInstr(desc)
                else:
                    self.instrs = OpenInstr(desc)
            else:
                if self._rand_bool():
                    self.instrs = GoToInstr(desc)
                else:
                    self.instrs = PickupInstr(desc)

# ----------------------------------------------------------------------

class FindObjS5MissionEnv(RoomGridLevel):
    def __init__(self, room_size=5, num_rows=3, num_cols=3, max_steps=None, **kwargs):
        if max_steps is None:
            max_steps = 20 * room_size**2
        self._forced_mission = None
        super().__init__(room_size=room_size, num_rows=num_rows, num_cols=num_cols, max_steps=max_steps, **kwargs)
        self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission: str):
        self._forced_mission = mission

    def gen_mission(self):
        target_obj = None
        if self._forced_mission is not None:
            m = re.match(r"pick up (?:a|an|the) (?:(\w+)\s)?(\w+)?", self._forced_mission)
            if m:
                color, obj_type = m.groups()
                if color in ("ball", "box", "key"):
                    obj_type = color
                    color = None
                
                i = self._rand_int(0, self.num_rows)
                j = self._rand_int(0, self.num_cols)
                target_obj, _ = self.add_object(i, j, kind=obj_type, color=color)
                self.place_agent(1, 1)
                self.connect_all()
                self.instrs = PickupInstr(ObjDesc(target_obj.type, target_obj.color))
            else:
                raise ValueError(f"Failed to parse forced mission: {self._forced_mission}")
        
        if target_obj is None:
            # Fallback only for initial environment creation
            i = self._rand_int(0, self.num_rows)
            j = self._rand_int(0, self.num_cols)
            target_obj, _ = self.add_object(i, j)
            self.place_agent(1, 1)
            self.connect_all()
            self.instrs = PickupInstr(ObjDesc(target_obj.type))


class ConstrainedPickupDistEnv(PickupDistMissionEnv):
    """PickupDist + random hazard tiles + per-step cost signal.

    The mission is e.g. "pick up the red box and avoid lava".
    On every step, info['tile_index'] gives the tile type under the agent
    and info['cost'] is 1.0 if that tile matches the constraint hazard.
    """

    def __init__(self, room_size=8, num_dists=2, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []  # set by set_forced_mission
        super().__init__(room_size=room_size, num_dists=num_dists, **kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')


    def set_forced_mission(self, mission):
        # Parse "pick up the red box and avoid lava and avoid water"
        super().set_forced_mission(mission)
        self.constraint_tiles = []
        for hazard, idx in HAZARD_TYPES.items():
            if f"avoid {hazard}" in mission:
                self.constraint_tiles.append(idx)

    def gen_mission(self):
        super().gen_mission()  # places goal + distractors
        
        # Place hazard classes that correspond to the constraints
        for constraint_tile in self.constraint_tiles:
            hazard_cls = None
            for name, idx in HAZARD_TYPES.items():
                if idx == constraint_tile:
                    hazard_cls = HAZARD_CLASSES[name]
                    break
            
            if hazard_cls is not None:
                # Find all valid empty cells
                empty_cells = []
                for i in range(self.grid.width):
                    for j in range(self.grid.height):
                        if (self.grid.get(i, j) is None
                                and not np.array_equal(self.agent_pos, (i, j))):
                            empty_cells.append((i, j))
                
                # Calculate how many to place (based on density), cap it at 4
                num_to_place = 0
                for _ in range(len(empty_cells)):
                    if self._rand_float(0, 1) < self.hazard_density:
                        num_to_place += 1
                
                num_to_place = min(num_to_place, self.max_hazards)
                
                # If we have cells and need to place some, randomly select them
                if num_to_place > 0 and empty_cells:
                    self.np_random.shuffle(empty_cells)
                    for pos in empty_cells[:num_to_place]:
                        self.grid.set(pos[0], pos[1], hazard_cls())

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        tile_idx = get_tile_index(cell)
        info['tile_index'] = tile_idx
        info['cost'] = 1.0 if tile_idx in self.constraint_tiles else 0.0
        # Hazards do not terminate episode in constraint setup
        if terminated and cell is not None and cell.type in ('lava', 'grass', 'water'):
            terminated = False
            reward = 0  
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# Shared hazard-scattering mixin used by all new constrained environments
# ──────────────────────────────────────────────────────────────────────

class _ConstrainedHazardMixin:
    """Mixin that adds hazard scattering + per-step cost signal.

    Sub-classes must:
      - Call `_parse_constraint(mission)` inside `set_forced_mission()`
      - Call `_scatter_hazards()` at the END of `gen_mission()`
      - Route `step()` through `_constrained_step()`
    """
    hazard_density: float = 0.15
    max_hazards: int = 2

    def _parse_constraint(self, mission: str):
        self.constraint_tiles = []
        for hazard, idx in HAZARD_TYPES.items():
            if f"avoid {hazard}" in mission:
                self.constraint_tiles.append(idx)

    def _scatter_hazards(self):
        if not self.constraint_tiles:
            return

        # For multi-room environments (GoToObjDoor, GoToOpen, OpenDoor, etc.),
        # restrict hazard placement to the center room only (room index 1,1)
        # so that lava never appears in unreachable outer rooms.
        if hasattr(self, 'get_room'):
            try:
                center_room = self.get_room(1, 1)
                rx, ry = center_room.top
                rw, rh = center_room.size
                # Exclude the outermost wall tiles (offset by 1)
                x_range = range(rx + 1, rx + rw - 1)
                y_range = range(ry + 1, ry + rh - 1)
            except Exception:
                x_range = range(self.grid.width)
                y_range = range(self.grid.height)
        else:
            x_range = range(self.grid.width)
            y_range = range(self.grid.height)

        for constraint_tile in self.constraint_tiles:
            hazard_cls = None
            for name, idx in HAZARD_TYPES.items():
                if idx == constraint_tile:
                    hazard_cls = HAZARD_CLASSES[name]
                    break
            if hazard_cls is None:
                continue

            empty_cells = [
                (i, j)
                for i in x_range
                for j in y_range
                if self.grid.get(i, j) is None
                and not np.array_equal(self.agent_pos, (i, j))
            ]
            num_to_place = min(
                sum(1 for _ in empty_cells if self._rand_float(0, 1) < self.hazard_density),
                self.max_hazards
            )
            if num_to_place > 0 and empty_cells:
                self.np_random.shuffle(empty_cells)
                for pos in empty_cells[:num_to_place]:
                    self.grid.set(pos[0], pos[1], hazard_cls())

    def _constrained_step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        tile_idx = get_tile_index(cell)
        info['tile_index'] = tile_idx
        info['cost'] = 1.0 if tile_idx in self.constraint_tiles else 0.0
        if terminated and cell is not None and cell.type in ('lava', 'grass', 'water'):
            terminated = False
            reward = 0
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# ConstrainedGoToObjDoorEnv
# Mission: "go to the <color> <obj/door> and avoid <hazard>"
# ──────────────────────────────────────────────────────────────────────

class ConstrainedGoToObjDoorEnv(GoToObjDoorMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)
        self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self.constraint_tiles = []
        for hazard, idx in HAZARD_TYPES.items():
            if f"avoid {hazard}" in mission:
                self.constraint_tiles.append(idx)

    def gen_mission(self):
        super().gen_mission()
        
        for constraint_tile in getattr(self, 'constraint_tiles', []):
            hazard_cls = None
            for name, idx in HAZARD_TYPES.items():
                if idx == constraint_tile:
                    hazard_cls = HAZARD_CLASSES[name]
                    break
            
            if hazard_cls is not None:
                if hasattr(self, 'get_room'):
                    try:
                        center_room = self.get_room(1, 1)
                        rx, ry = center_room.top
                        rw, rh = center_room.size
                        x_range = range(rx + 1, rx + rw - 1)
                        y_range = range(ry + 1, ry + rh - 1)
                    except Exception:
                        x_range = range(self.grid.width)
                        y_range = range(self.grid.height)
                else:
                    x_range = range(self.grid.width)
                    y_range = range(self.grid.height)

                empty_cells = []
                for i in x_range:
                    for j in y_range:
                        if (self.grid.get(i, j) is None
                                and not np.array_equal(self.agent_pos, (i, j))):
                            empty_cells.append((i, j))
                
                num_to_place = 0
                for _ in range(len(empty_cells)):
                    if self._rand_float(0, 1) < self.hazard_density:
                        num_to_place += 1
                
                num_to_place = min(num_to_place, self.max_hazards)
                
                if num_to_place > 0 and empty_cells:
                    self.np_random.shuffle(empty_cells)
                    for pos in empty_cells[:num_to_place]:
                        self.grid.set(pos[0], pos[1], hazard_cls())

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        tile_idx = get_tile_index(cell)
        info['tile_index'] = tile_idx
        info['cost'] = 1.0 if (hasattr(self, 'constraint_tiles')
                               and tile_idx in self.constraint_tiles) else 0.0
        
        if terminated and cell is not None and cell.type in ('lava', 'grass', 'water'):
            terminated = False
            reward = 0  
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────
# ConstrainedGoToOpenEnv
# Mission: "go to the <color> <obj/door> and avoid <hazard>"
# ──────────────────────────────────────────────────────────────────────

class ConstrainedGoToOpenEnv(_ConstrainedHazardMixin, GoToOpenMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')


    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def step(self, action):
        return self._constrained_step(action)


# ──────────────────────────────────────────────────────────────────────
# ConstrainedOpenDoorEnv
# Mission: "open the <color> door and avoid <hazard>"
# ──────────────────────────────────────────────────────────────────────

class ConstrainedOpenDoorEnv(_ConstrainedHazardMixin, OpenDoorMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')


    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def step(self, action):
        return self._constrained_step(action)


# ──────────────────────────────────────────────────────────────────────
# ConstrainedOpenDoorLocEnv
# Mission: "open the door on the <loc> and avoid <hazard>"
# ──────────────────────────────────────────────────────────────────────

class ConstrainedOpenDoorLocEnv(_ConstrainedHazardMixin, OpenDoorLocMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')

    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def step(self, action):
        return self._constrained_step(action)


# ──────────────────────────────────────────────────────────────────────
# ConstrainedOpenDoorsOrderEnv
# Mission: "open the <c1> door, then open the <c2> door and avoid <hazard>"
# ──────────────────────────────────────────────────────────────────────

class ConstrainedOpenDoorsOrderEnv(_ConstrainedHazardMixin, OpenDoorsOrderMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)
        # self.render_mode = kwargs.get('render_mode', 'human')


    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def step(self, action):
        return self._constrained_step(action)




class ConstrainedActionObjDoorEnv(_ConstrainedHazardMixin, ActionObjDoorMissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)

    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def step(self, action):
        return self._constrained_step(action)

class ConstrainedFindObjS5Env(_ConstrainedHazardMixin, FindObjS5MissionEnv):
    def __init__(self, hazard_density=0.15, max_hazards=2, **kwargs):
        self.hazard_density = hazard_density
        self.max_hazards = max_hazards
        self.constraint_tiles = []
        super().__init__(**kwargs)

    def set_forced_mission(self, mission):
        goal_part = mission.split(" and avoid ")[0].strip()
        super().set_forced_mission(goal_part)
        self._parse_constraint(mission)

    def gen_mission(self):
        super().gen_mission()
        self._scatter_hazards()

    def _scatter_hazards(self):
        # Override to scatter across the entire 3x3 grid of rooms for FindObjS5
        if not self.constraint_tiles:
            return

        x_range = range(self.grid.width)
        y_range = range(self.grid.height)

        for constraint_tile in self.constraint_tiles:
            hazard_cls = None
            for name, idx in HAZARD_TYPES.items():
                if idx == constraint_tile:
                    hazard_cls = HAZARD_CLASSES[name]
                    break
            if hazard_cls is None:
                continue

            empty_cells = [
                (i, j)
                for i in x_range
                for j in y_range
                if self.grid.get(i, j) is None
                and not np.array_equal(self.agent_pos, (i, j))
            ]
            num_to_place = min(
                sum(1 for _ in empty_cells if self._rand_float(0, 1) < self.hazard_density),
                self.max_hazards
            )
            if num_to_place > 0 and empty_cells:
                self.np_random.shuffle(empty_cells)
                for pos in empty_cells[:num_to_place]:
                    self.grid.set(pos[0], pos[1], hazard_cls())

    def step(self, action):
        return self._constrained_step(action)
