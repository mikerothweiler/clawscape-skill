"""The atlas only compounds if its guarantees hold, so they are tested.

These are less about atlas.py being correct and more about it staying wired in.
Knowledge that depends on somebody remembering a step is knowledge that gets
lost; these tests fail loudly if a refactor quietly removes the recording, or
if the rules that stop bad knowledge entering the map are weakened.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes"
    ),
)

import atlas  # noqa: E402

RECIPES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes"
)


class ObservationIsWiredIn(unittest.TestCase):
    """The walkers must record as a side effect of walking.

    If someone refactors these and drops the call, sessions silently stop
    learning and nobody notices for weeks.
    """

    def test_walk_observes(self):
        src = open(os.path.join(RECIPES, "walk.py")).read()
        self.assertIn(
            "atlas.observe(",
            src,
            "walk.py must observe; without it walking teaches nothing",
        )

    def test_maze_observes(self):
        src = open(os.path.join(RECIPES, "maze.py")).read()
        self.assertIn(
            "atlas.observe(",
            src,
            "maze.py must observe; its refusals are the blocked map",
        )

    def test_maze_records_why_a_tile_refused(self):
        src = open(os.path.join(RECIPES, "maze.py")).read()
        self.assertIn(
            "reason=", src, "a refusal without a reason cannot be told from a wall"
        )


class RefusalIsNotAlwaysAWall(unittest.TestCase):
    """The rules that keep wrong knowledge out of the shared map."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.atlas = os.path.join(self.dir, "atlas.json")
        self.obs = os.path.join(self.dir, "obs.jsonl")

    def _observe(self, **kw):
        atlas.observe(
            kw.pop("state", {"player": {"worldX": 3100, "worldZ": 3500}}),
            path=self.obs,
            **kw,
        )

    def _fold(self):
        return atlas.fold(obs_path=self.obs, atlas_path=self.atlas)

    def test_one_refusal_is_not_believed(self):
        self._observe(refused=[3110, 3510], reason="something stood there")
        self._fold()
        a = atlas._load(self.atlas)
        self.assertNotIn(
            "3110,3510",
            atlas.blocked(a),
            "a single refusal must not poison the map for everyone",
        )

    def test_repeated_refusals_are_believed(self):
        for _ in range(atlas.BLOCK_CONFIDENCE):
            self._observe(refused=[3111, 3511], reason="terrain")
            self._fold()
        a = atlas._load(self.atlas)
        self.assertIn("3111,3511", atlas.blocked(a))

    def test_standing_on_a_tile_clears_its_block(self):
        for _ in range(atlas.BLOCK_CONFIDENCE):
            self._observe(refused=[3112, 3512], reason="terrain")
            self._fold()
        self.assertIn("3112,3512", atlas.blocked(atlas._load(self.atlas)))
        self._observe(stood=[3112, 3512])
        self._fold()
        self.assertNotIn(
            "3112,3512",
            atlas.blocked(atlas._load(self.atlas)),
            "direct evidence must always beat an inferred block",
        )

    def test_walked_tile_never_becomes_blocked(self):
        self._observe(stood=[3113, 3513])
        self._fold()
        for _ in range(5):
            self._observe(refused=[3113, 3513], reason="monster in the way")
            self._fold()
        self.assertNotIn("3113,3513", atlas.blocked(atlas._load(self.atlas)))


class RequirementGatedPassagesAreNotWalls(unittest.TestCase):
    """The Pirates' Hideout door needs Thieving 39 and a lockpick.

    Recording it as impassable would tell every future agent to give up on a
    door that simply wants a level and an item.
    """

    def test_shipped_atlas_records_the_door_as_a_crossing(self):
        a = atlas._load(os.path.join(RECIPES, "atlas.json"))
        door = a["crossings"].get("2558")
        self.assertIsNotNone(door, "the hideout door must be a known crossing")
        self.assertIn("requires", door, "a gated passage must say what it requires")
        self.assertNotIn("2558", atlas.blocked(a))

    def test_blocked_never_includes_a_crossing(self):
        a = atlas._load(os.path.join(RECIPES, "atlas.json"))
        self.assertTrue(set(a["crossings"]).isdisjoint(atlas.blocked(a)))


class TheDictionaryKnowsTheNamelessThings(unittest.TestCase):
    """The gate that cost a day has no name in the world's own data."""

    def test_nameless_wilderness_gates_have_aliases(self):
        a = atlas._load(os.path.join(RECIPES, "atlas.json"))
        for oid in ("1596", "1597"):
            o = a["objects"].get(oid)
            self.assertIsNotNone(o, f"loc {oid} must be in the dictionary")
            self.assertTrue(
                o.get("alias"),
                f"loc {oid} is nameless in loc.pack and must carry "
                "an alias, or it is undiscoverable by search",
            )

    def test_crossing_verbs_are_data_not_prose(self):
        a = atlas._load(os.path.join(RECIPES, "atlas.json"))
        verbs = {c.get("verb") for c in a["crossings"].values()}
        for verb in ("Open", "Cross", "Slash"):
            self.assertIn(
                verb,
                verbs,
                "a walker must be able to enumerate crossing verbs "
                "rather than probing only for Open",
            )


class ObservingNeverBreaksPlay(unittest.TestCase):
    def test_bad_state_is_swallowed(self):
        atlas.observe(None)
        atlas.observe({}, path=os.path.join(tempfile.mkdtemp(), "o.jsonl"))

    def test_unwritable_log_does_not_raise(self):
        atlas.observe(
            {"player": {"worldX": 1, "worldZ": 1}},
            path="/definitely/not/a/path/o.jsonl",
        )


class TheAtlasStaysReadable(unittest.TestCase):
    def test_shipped_atlas_parses_and_has_content(self):
        a = json.load(open(os.path.join(RECIPES, "atlas.json")))
        self.assertGreater(len(a["tiles"]), 100)
        self.assertGreater(len(a["objects"]), 0)


if __name__ == "__main__":
    unittest.main()


class BadReadsStayOutOfTheMap(unittest.TestCase):
    """Tile "1,1" once reached the shared atlas as confirmed walkable ground.

    A planner cannot tell a phantom tile from a real one, so a single malformed
    state read becomes a permanent wrong entry in every agent's map.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.atlas = os.path.join(self.dir, "atlas.json")
        self.obs = os.path.join(self.dir, "obs.jsonl")

    def test_out_of_world_tile_is_not_recorded(self):
        atlas.observe(
            {"player": {"worldX": 1, "worldZ": 1}}, stood=[1, 1], path=self.obs
        )
        atlas.fold(obs_path=self.obs, atlas_path=self.atlas)
        self.assertNotIn("1,1", atlas._load(self.atlas).get("tiles", {}))

    def test_high_z_regions_are_real_and_kept(self):
        """z runs to 10367 here. Treating high z as junk would delete real map."""
        self.assertTrue(atlas.in_world([3243, 9893]))
        self.assertTrue(atlas.in_world([3093, 3518]))

    def test_obvious_nonsense_is_rejected(self):
        for bad in ([1, 1], [0, 0], [-5, 3000], [99999, 3000], None, ["a", "b"]):
            self.assertFalse(atlas.in_world(bad) if bad else False)

    def test_shipped_atlas_has_no_out_of_world_tiles(self):
        a = atlas._load(os.path.join(RECIPES, "atlas.json"))
        bad = [t for t in a.get("tiles", {}) if not atlas.in_world(t.split(","))]
        self.assertEqual(bad, [], "the shared atlas must contain only real tiles")


class TerrainIsReadable(unittest.TestCase):
    """The MAP section carries per-tile blocking flags.

    Nothing read them until 2026-09-13, so every offline plan routed straight
    through water and lava and only found out by being refused live.
    """

    def setUp(self):
        sys.path.insert(0, RECIPES)
        import mapdata

        self.mapdata = mapdata

    def test_blocked_bit_is_one(self):
        self.assertEqual(self.mapdata.BLOCKED_BIT, 1)

    def test_terrain_blocked_is_unioned_into_blocked(self):
        src = open(os.path.join(RECIPES, "mapdata.py")).read()
        self.assertIn(
            "terrain_blocked(",
            src.split("def blocked(")[1],
            "blocked() must include terrain, or plans route through water",
        )

    def test_passable_things_are_never_blocking(self):
        """A door matching a blocking keyword must still be passable."""
        for name in ("inaccastledoubledoorropen", "openbankdoor_l", "wildernessgate"):
            low = name.lower()
            blocks = any(k in low for k in self.mapdata.BLOCKING)
            passes = any(p in low for p in self.mapdata.PASSABLE)
            if blocks:
                self.assertTrue(
                    passes, f"{name} matches a blocking keyword and must be excused"
                )


class WallsSitOnEdgesNotTiles(unittest.TestCase):
    """A wall loc occupies a tile EDGE, which no tile-level model can express.

    Measured live: a character at (2618,3315) could not reach a ladder one tile
    west at (2617,3315). The LOC row on her own tile is `1602 0` -- a
    timberwall, shape 0, rotation 0, a wall on her west edge. Both tiles are
    perfectly standable, so "which tiles are blocked" can never answer it.
    """

    def setUp(self):
        sys.path.insert(0, RECIPES)
        import mapdata

        self.mapdata = mapdata

    def test_rotation_maps_to_the_four_edges(self):
        self.assertEqual(
            set(self.mapdata._EDGE.values()), {(-1, 0), (0, 1), (1, 0), (0, -1)}
        )

    def test_rotation_zero_is_west(self):
        """Rotation is omitted from a LOC row when it is 0, so 0 must be real."""
        self.assertEqual(self.mapdata._EDGE[0], (-1, 0))

    def test_ground_decor_never_blocks(self):
        self.assertIn(22, self.mapdata.IGNORED_SHAPES)

    def test_centrepieces_occupy_whole_tiles(self):
        for shape in (9, 10, 11):
            self.assertIn(shape, self.mapdata.SOLID_SHAPES)


class TravelRecipesMustRecord(unittest.TestCase):
    """Only hops written to routes.json feed route.py, the proven-roads planner.

    walk.leg() returns hops but does not write them -- the writing lives in
    walk.py's main(). So any recipe calling leg() directly walks a long way and
    teaches nobody, which is how a character crossed half the world and still
    got `off_the_map` from route.py.
    """

    def test_trek_records_hops(self):
        src = open(os.path.join(RECIPES, "trek.py")).read()
        self.assertIn(
            "record(",
            src,
            "trek.py calls walk.leg() directly, so it must record hops itself",
        )

    def test_walk_records_hops(self):
        src = open(os.path.join(RECIPES, "walk.py")).read()
        self.assertIn("record(", src)

    def test_route_reads_the_atlas(self):
        """The atlas is the richest record of ground actually stood on."""
        src = open(os.path.join(RECIPES, "route.py")).read()
        self.assertIn("add_atlas_tiles", src)
