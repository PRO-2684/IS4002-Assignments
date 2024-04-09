from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch

# BEGIN_YOUR_CODE (You may add some codes here to assist your coding below if you want, but don't worry if you deviate from this.)

# END_YOUR_CODE

# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(location=self.startLocation)
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        currentTags = self.cityMap.tags[state.location]
        return self.endTag in currentTags
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        ret = []
        successors = self.cityMap.distances[state.location]

        for next, dist in successors.items():
            ret.append((next, State(location=next), dist))
            
        return ret
        # END_YOUR_CODE


########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. If you prefer, you may create a new map using via
    `createCustomMap()`.

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "park", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Or, if you would rather use a custom map, you can uncomment the following!
    # cityMap = createCustomMap("data/custom.pbf", "data/custom-landmarks".json")

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = "9898608898"
    endTag = "amenity=parking_entrance"
    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags)) # Remaining tags to visit

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        startTags = self.cityMap.tags[self.startLocation]
        memory = tuple(bool(tag in startTags) for tag in self.waypointTags)
        return State(location=self.startLocation, memory=memory)
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        currentTags = self.cityMap.tags[state.location]
        if self.endTag not in currentTags: # If we haven't reached the end
            return False
        return all(state.memory) # If we have visited all waypoints
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
        res = []
        successors = self.cityMap.distances[state.location]

        for next, dist in successors.items():
            nextMemory = list(state.memory)
            nextTags = self.cityMap.tags[next]
            for i in range(len(self.waypointTags)):
                nextMemory[i] |= (self.waypointTags[i] in nextTags)
            nextMemory = tuple(nextMemory)
            res.append((next, State(location=next, memory=nextMemory), dist))

        return res
        # END_YOUR_CODE

########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    startLocation = "4159990495"
    waypointTags = [makeTag("direction", "forward"), makeTag("direction", "backward"), makeTag("direction", "anticlockwise")]
    endTag="landmark=AOERC"
    # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.startState()
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.isEnd(state)
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
            res = []
            origSuccessors = problem.successorsAndCosts(state)

            for next, nextState, cost in origSuccessors:
                newCost = cost + heuristic.evaluate(nextState) - heuristic.evaluate(state)
                res.append((next, nextState, newCost))

            return res
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        self.destinations = tuple(filter(
            lambda dest: endTag in self.cityMap.tags[dest],
            self.cityMap.geoLocations.keys()
        )) # Filter out destinations that don't have the endTag
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        return max(map(
            lambda dest: computeDistance(
                self.cityMap.geoLocations[state.location], self.cityMap.geoLocations[dest]
            ),
            self.destinations
        ))
        # END_YOUR_CODE


########################################################################################
# Problem 3c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """

    def __init__(self, endTag: str, cityMap: CityMap):
        # Precompute
        # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
        class ShortestPathProblemRev(SearchProblem):
            def startState(self) -> State:
                return State("NON-EXISTENT")

            def isEnd(self, state: State) -> bool:
                return False  # Make it search for all

            def successorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                ret = []
                if state.location == "NON-EXISTENT":
                    for dest in cityMap.geoLocations.keys():
                        if endTag in cityMap.tags[dest]:
                            ret.append((dest, State(dest), 0))
                else:
                    successors = cityMap.distances[state.location]
                    for next, dist in successors.items():
                        ret.append((next, State(next), dist))
                return ret

        problem = ShortestPathProblemRev()
        search = UniformCostSearch()
        search.solve(problem)
        self.heuristics = search.pastCosts
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.heuristics[state.location]
        # END_YOUR_CODE
