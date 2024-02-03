# pylint: disable=too-many-lines
"""
Module for building and running the initial proof-of-concept simulation.
"""

from abc import ABC, abstractmethod
import random

import geopandas as gpd
import networkx as nx
import simpy

random.seed(666)  # randomize with the beast

# TODO (@ps-jbunton): Use pint package for more careful time unit management.
# TODO (@ps-jbunton): Consider using dataclasses as needed.


class PayloadLocation(ABC):
    """
    Abstract base class for all possible locations that Payloads can be.
    """


class Payload:
    """
    Payload class.  Typically shipping containers, these represent objects that can be delivered via
    trucks or ParallelVehicles.


    Attributes
    ----------
    env: simpy.Environment
        The simpy simulation environment the Payload exists in.
    destination: PayloadLocation
        GeoLocation that the payload needs to be delivered.
    current_location: PayloadLocation
        The current location of the payload.
    origin: PayloadLocation
        Where the payload started its life.
    """

    def __init__(
        self,
        env: simpy.Environment,
        destination: PayloadLocation,
        current_location: PayloadLocation,
    ) -> None:
        """
        Constructs a new Payload object.

        Parameters
        ----------
        env: simpy.Environment
            The simpy simulation environment for the payload.
        destination: PayloadLocation
            The payload's desired destination.
        current_location: ParallelVehicle | ContainerYard | RailTerminal
            Where to spawn the container.
        """
        self.env = env
        self.destination = destination
        self.current_location = current_location
        self.origin = current_location

    def get_size(self):
        """
        Computes the `size` of the Payload.  Right now, every Payload is just
        one unit, but we could make this more complicated.
        """
        return 1

    def set_location(self, location: PayloadLocation):
        """
        Helper setting function for the location of the Payload.
        """
        self.current_location = location

    def get_location(self):
        """
        Helper getter function for location of the Payload.
        """
        return self.current_location

    def __format__(self, spec) -> str:
        """
        Convenience method for printing out details of a Payload.
        """
        return f"Payload at {self.current_location}\nGoing to: {self.destination}"


class Vehicle(PayloadLocation):
    """
    Abstract base class for all vehicles that can carry Payloads.
    """

    @abstractmethod
    def payload_capacity(self) -> int:
        """
        Report how many Payloads can be loaded onto this Vehicle.
        """

    @abstractmethod
    def current_payload_count(self) -> int:
        """
        Report how many Payloads the vehicle has.
        """

    @abstractmethod
    def get_payload(self):
        """
        Retrieve a loaded Payload from the Vehicle (pops from the end of the list, so LIFO), or None
        if there are none to retrieve.
        """

    @abstractmethod
    def put_payload(self, payload: Payload):
        """
        Put a Payload onto the Vehicle (appends to the list).
        """


class GeoLocation:
    """
    Physical location convenience class.

    Attributes
    ----------
    _latitude: float
        Latitude of the location.
    _longitude: float
        Longitude of the location.
    _rail_node_id: int
        An optional identifier for which node ID in a RailNetwork to associate with.
    _address: str | None
        String format of the address (for querying in Google Maps, for example)
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        rail_node_id: int | None = None,
        address: str | None = None,
    ) -> None:
        """
        Constructs a new GeoLocation.

        Parameters
        ----------
        latitude: float
            GeoLocation's latitude coordinate.
        longitude: float
            GeoLocation's longitude coordinate.
        address: str, optional
            String with street address of location.
        """
        self._latitude = latitude
        self._longitude = longitude
        self._address = address
        self._rail_node_id = rail_node_id

    def __format__(self, spec) -> str:
        """
        Convenience method for printing out the details of a GeoLocation.
        """
        if self._address is not None:
            s = f"{self._address}"
        else:
            s = f"GeoLocation at ({self._latitude:.6f}, {self._longitude:.6f})"
        if self._rail_node_id is not None:
            s += f"\nwith rail node ID: {self._rail_node_id}"
        return s

    def get_latitude(self) -> float:
        """
        Getter for location latitude.
        """
        return self._latitude

    def get_longitude(self) -> float:
        """
        Getter for location longitude.
        """
        return self._longitude

    def get_latitude_and_longitude(self) -> tuple[float, float]:
        """
        Getter for latitude and longitude pair, returned as a tuple of floats (latitude, longitude).
        """
        return (self._latitude, self._longitude)

    def get_rail_node_id(self) -> int | None:
        """
        Getter for rail ID.  If there isn't one, returns None.
        """
        return self._rail_node_id


class RailTerminal(PayloadLocation):
    """
    Rail terminal class.  Models the railyard endpoints for routes, which may have
    `ParallelVehicle`s and `Truck`s enter it and move the `Payload`s through its
    `arrival_queue` and `departure_queue` simpy Stores.

    Attributes
    ----------
    env: simpy.Environment
        The simpy environment the RailTerminal exists in.
    _location: GeoLocation | None
        The physical location to associate with this terminal, if any.
    arrival_queue: simpy.FilterStore
        A simpy Store object that holds Payloads that were brought to the terminal and are waiting
        to be loaded onto a rail vehicle.
    departure_queue: simpy.FilterStore
        A simpy Store object that holds Payloads brought to the terminal by rail vehicles that are
        waiting to be picked up by a road vehicle for delivery.
    name: str
        An identifying name for the rail terminal.
    reporting_list: list[dict[str, value]]
        A list of reported statuses or events, where the elements are dicts
        keyed by strings representing the quantity and values representing the
        reported status/quantity.
    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        location: GeoLocation | None = None,
    ) -> None:
        """
        Constructs a new rail terminal object.

        Parameters
        ----------
        env: simpy.Environment
            The simpy environment the rail terminal will exist in.
        name: str
            An identifying name for the rail terminal, i.e., "Savannah" or "Cordele"
        location: GeoLocation, optional
            An optional geographic location for the terminal to exist at.
        """
        super().__init__()  # This is just so I can register the object as a PayloadLocation
        self.env = env
        self.name = name
        self._location = location
        self.reporting_list: list[dict[str, float]] = []

        # Build the terminal's arrival and departure queues, use infinite capacity for now.
        self.arrival_queue = simpy.FilterStore(self.env)
        self.departure_queue = simpy.FilterStore(self.env)

    def __format__(self, spec) -> str:
        """
        Convenience function for writing out details of RailTerminal.
        """
        s = f"Name: {self.name}\n"
        if self._location:
            s += f"Location: {self._location}\n"
        s += f"arrival payload count: {len(self.arrival_queue.items)}\n"
        s += f"departure payload count: {len(self.departure_queue.items)}"
        return s

    def process_incoming_rail_vehicle(self, vehicle: Vehicle):
        """
        Process an incoming rail vehicle through the RailTerminal.  This will unload all Payloads on
        the object (regardless of their destination) and put them in the RailTerminal's
        arrival_queue.
        """

        # While there are payloads on the vehicle, unload them into the arrivals queue.
        while vehicle.current_payload_count() > 0:
            payload = vehicle.get_payload()
            if payload is not None:
                yield from self.put_in_arrival_queue(payload)

        # While there is space in the vehicle and we have outbound Payloads in our departure queue,
        # load them onto the vehicle.
        while (
            vehicle.current_payload_count() < vehicle.payload_capacity()
            and self.departure_queue.items
        ):
            # Note that we do not apply any payload filtering right now, but we could.
            payload = yield from self.get_from_departure_queue()
            vehicle.put_payload(payload)

    def put_in_arrival_queue(self, payload: Payload):
        """
        Convenience function to put a Payload object in the arrivals queue,
        then report the updated state.
        """
        yield self.arrival_queue.put(payload)
        self._report_current_state()

    def put_in_departure_queue(self, payload: Payload):
        """
        Convenience function to put a Payload object in the departures queue,
        then report the updated state.
        """
        yield self.departure_queue.put(payload)
        self._report_current_state()

    def get_from_arrival_queue(self, filter_fn=lambda _: True) -> Payload:
        """
        Convenience function to request a Payload object from the arrivals
        queue. Once one is available matching the given filter_fn, it is
        returned and the updated state is reported.
        """
        payload = yield self.arrival_queue.get(filter_fn)
        self._report_current_state()
        return payload

    def get_from_departure_queue(self, filter_fn=lambda _: True):
        """
        Convenience function to request a Payload object from the departures
        queue. Once one is available matching the given filter_fn, it is
        returned and the updated state is reported.
        """
        payload = yield self.departure_queue.get(filter_fn)
        self._report_current_state()
        return payload

    def get_location(self) -> GeoLocation | None:
        """
        Getter function for retrieving the RailTerminal's assigned GeoLocation, returning None if
        there is not one (since the default value is None).
        """
        return self._location

    def _report_current_state(self) -> None:
        """
        Internal method that requests the current object state be attached to
        the running report_list.
        """
        self.reporting_list.append(self._get_timestamped_state())

    def _get_timestamped_state(self):
        """
        Internal method that requests the current object state, formatted as a
        dict keyed by strings denoting state variables mapping to the
        corresponding state values.
        """
        return {
            "TIME": self.env.now,
            "ARRIVAL_QUEUE_SIZE": len(self.arrival_queue.items),
            "DEPARTURE_QUEUE_SIZE": len(self.departure_queue.items),
        }


class ContainerYard(PayloadLocation):
    """
    Container yard class.  Models Container Yards, which request `Payload`s and receives them.

    Attributes
    ----------
    env: simpy.Environment
        The simpy environment the container yard lives in.
    id: int
        Unique int for identifying this ContainerYard from others.
    requests_per_day: float
        Average number of requests for Payloads that the container yard makes per day.  Requests
        will be generated using a Poisson process with rate parameter requests_per_day
    payload_sink: simpy.Container
        Simpy container that represents delivered payloads to this container yard.
    name: str
        Name of the container yard.
    location: GeoLocation | None
        Physical geographical location info for the container yard.
    reporting_list: list[dict[str, value]]
        A list of reported statuses or events, where the elements are dicts
        keyed by strings representing the quantity and values representing the
        reported status/quantity.
    """

    def __init__(
        self,
        env: simpy.Environment,
        requests_per_day: float,
        name: str,
        location: GeoLocation | None = None,
    ) -> None:
        """
        Creates a new ContainerYard.

        Parameters
        ----------
        env: simpy.Environment
            The simpy environment the new container yard will exist in.
        requests_per_day: float
            Average number of requests for new payloads this container yard makes per day.
        name: str
            Name of the container yard.
        location: GeoLocation, optional
            GeoLocation associated to this container yard, if any.
        """
        self.env = env
        self.requests_per_day = requests_per_day
        self.name = name
        self.location = location
        self.payload_sink = simpy.Container(self.env)
        self.reporting_list: list[dict[str, float]] = []

    def __format__(self, spec) -> str:
        return (
            f"Name: {self.name}\nLocation: {self.location}\n"
            + f"Number of received containers:{self.payload_sink.level}"
        )

    def run(self, origin: RailTerminal):
        """
        Convenience class for running the container yard logic.
        """
        while True:
            yield from self.new_order(origin)

    def new_order(self, origin: RailTerminal):
        """
        Function that generates a new Payload at the given origin whose destination is this
        ContainerYard at the rate self.requests_per_day.

        Parameters
        ----------
        origin: RailTerminal
            Where the newly created Payload instance should be generated.
        """
        # Wait for a randomly generated amount of time obeying a Poisson process.
        time_until_next_order_hrs = self._time_before_new_request_hrs()
        new_order = yield self.env.timeout(
            delay=time_until_next_order_hrs,
            value=Payload(self.env, self, origin),
        )
        # Put the new order in the wait queue of its origin.
        origin.departure_queue.put(new_order)
        print(
            f"T = {float(self.env.now):.03f} | "
            + f"Container yard {self.name} generated a new order at {new_order.origin.name}."
        )

    def deliver_order(self, payload: Payload):
        """
        Convenience wrapper that takes a payload and records its unit in the
        payload_sink, then reports the new state.
        """
        # Put one payload in the sink
        yield self.payload_sink.put(payload.get_size())
        self._report_current_state()

    def _report_current_state(self):
        """
        Internal method for adding the current state dictionary to the report_list.
        """
        self.reporting_list.append(self._get_timestamped_state())

    def _get_timestamped_state(self):
        """
        Internal method for reporting all state variables and values in a dict.
        """
        return {"TIME": self.env.now, "RECEIVED": self.payload_sink.level}

    def _time_before_new_request_hrs(self):
        """
        Method that computes a new random time until the next request (in hours).  Assumes requests
        obey a Poisson arrival process, meaning they arrive with exponentially distributed
        """
        return random.expovariate(self.requests_per_day / 24.0)


class RailNetwork:
    """
    Class that encapsulates a railroad network.

    Attributes:
    -----------
    rail_network_graph : nx.Graph
        A networkX graph object that represents the rail network information.
    _nodes_df: gpd.GeoDataFrame
        A GeoDataFrame containing all of the network's information for the nodes of the network.
    _edges_df: gpd.GeoDataFrame
        A GeoDataFrame containing all of the network's information for the edges of the network.
    _edges_weight_col: str | int
        The column of the _edges_df GeoDataFrame containing the weight or length of the edges.
    """

    def __init__(
        self,
        edges_df: gpd.GeoDataFrame,
        edges_source_id_col: str | int,
        edges_target_id_col: str,
        edges_weight_col: str | int,
        edge_tag_cols: list[str | int],
        nodes_df: gpd.GeoDataFrame,
        node_id_col: str | int,
        node_tag_cols: list[str | int],
    ):
        """
        Parses data in GeoDataFrame format and creates a tagged nx.Graph object.

        Parameters:
        -----------
        edges_df: gpd.GeoDataFrame
            GeoDataFrame containing the network edge information.
        edges_source_id_col: str | int
            Column of the GeoDataFrame edges_df that specifies an edge's source node ID.
        edges_target_id_col: str | int
            Column of the GeoDataFrame edges_df that specifies an edge's target node ID.
        edges_weight_col: str | int
            Column of the GeoDataFrame edges_df that specifies an edge's weight/distance.
        edge_tag_cols: list[str | int]
            A list of all other columns in edges_df that should be added to the resulting nx.Graph()
            edge objects.
        nodes_df: gpd.GeoDataFrame
            GeoDataFrame contianing the network node information.
        node_id_col: str | int
            Column of nodes_df that contains the node ID referenced in edges_df.
        node_tag_cols: list[str | int]
            A list of all columns in nodes_df that should be added to the resulting nx.Graph() node
            objects.
        """
        self._nodes_df = nodes_df
        self._edges_df = edges_df
        self._edges_weight_col = edges_weight_col

        # This function does most of the heavy lifting for us
        self.rail_network_graph = nx.from_pandas_edgelist(
            df=edges_df,
            source=edges_source_id_col,
            target=edges_target_id_col,
            edge_attr=edge_tag_cols + [edges_weight_col],
        )
        for _, attrs in nodes_df.iterrows():
            nx.set_node_attributes(
                self.rail_network_graph, {attrs[node_id_col]: attrs[node_tag_cols]}
            )

    def compute_route(
        self,
        start_node_id: int,
        end_node_id: int,
        weight_col: str | int | None = None,
    ) -> list[int]:
        """
        Computes the sequence of rail nodes to traverse to get from the designated start to finish.
        Currently uses the shortest path, with weights determined by the weight_col attribute of
        each edge in the network graph.
        """
        # If asked for a path starting and ending at the same place, don't move.
        if start_node_id == end_node_id:
            return []

        weights = weight_col or self._edges_weight_col
        return nx.shortest_path(
            G=self.rail_network_graph,
            source=start_node_id,
            target=end_node_id,
            weight=weights,
        )

    def register_terminal_at_node(
        self,
        terminal: RailTerminal,
        node_id: int,
    ):
        # TODO (@ps-jbunton): Think about whether this is the best solution for network-entity
        # mappings.
        """
        Register the given RailTerminal as being at the given node_id in the graph. Under the hood,
        this makes the RailTerminal an attribute of the rail_network nx.Graph.  The nx.Graph
        attribute should be a reference to the original RailTerminal, meaning only one instance
        exists and mutation can occcur externally or through the nx.Graph.
        """
        self.rail_network_graph.nodes[node_id]["Terminal"] = terminal

    def check_for_rail_terminal(self, node_id: int) -> RailTerminal | None:
        """
        Given a node ID, checks if there is a RailTerminal associated with that node in the
        rail_network_graph.  If so, returns it, otherwise, returns None.
        """
        if "Terminal" in self.rail_network_graph.nodes[node_id]:
            terminal = self.rail_network_graph.nodes[node_id]["Terminal"]
            if isinstance(terminal, RailTerminal):
                return terminal
        # If there isn't one logged there, or it isn't a terminal, return None
        return None

    def get_segment_length_m(
        self, node_id1: int, node_id2: int, weight_col: int | str | None = None
    ):
        """
        Helper function for getting how long a segment of track connecting two nodes is.
        """
        weight = weight_col or self._edges_weight_col
        assert self.rail_network_graph.has_edge(
            node_id1, node_id2
        ), "Edge is not in graph--segment does not exist!"
        return self.rail_network_graph[node_id1][node_id2][weight]

    def get_node_lat_lon(self, node_id: int):
        """
        Given a node id in the current rail_network_graph, extract the latitude and longitude of the
        node.
        """
        raise NotImplementedError

    def find_nearest_node_from_lat_lon(self, latitude: float, longitude: float) -> int:
        """
        Given a latitude and longitude pair, finds the closest points.
        """
        raise NotImplementedError("Not implemented yet!")

    def node_attrs(self, node_id):
        """
        Convenience getter for node attribute dictionary at a given node id.
        """
        return self.rail_network_graph.nodes[node_id]


class ParallelVehicle(Vehicle):
    """
    Parallel vehicle class.  Models Parallel Vehicles as they move `Payload`s
    through the system.

    Attributes
    ----------
    env: simpy.Environment
        The simpy simulation environment the vehicle exists in.
    vehicle_id: int
        Unique int for identifying this ParallelVehicle from others.
    capacity: int
        How many `Payload`s can be loaded on the vehicle.
    rail_network: RailNetwork
        The RailNetwork the vehicle is currently operating in.
    current_node_id: int
        The node ID in the rail_network that the vehicle is currently located at.
    destination: RailTerminal | None
        The terminal the vehicle is headed to, or None if vehicle is not
        preparing to move.
    current_payload: List[Payload]
        List containing the Payload objects that the vehicle currently has
        loaded.
    current_terminal: RailTerminal | None
        The current RailTerminal the vehicle is in, if any.
    reporting_list: list[dict[str, value]]
        A list of reported statuses or events, where the elements are dicts
        keyed by strings representing the quantity and values representing the
        reported status/quantity.
    current_velocity_ms: float
        The current velocity of the vehicle.
    """

    def __init__(
        self,
        env: simpy.Environment,
        vehicle_id: int,
        capacity: int,
        rail_network: RailNetwork,
        current_rail_node_id: int,
        current_route: list[int] | None = None,
        destination: RailTerminal | None = None,
        current_payload: list[Payload] | Payload | None = None,
    ) -> None:
        """
        Constructs a new ParallelVehicle.

        Parameters
        ----------
        env: simpy.Environment
            The simpy environment the vehicle will exist in.
        vehicle_id: int
            An int for identifying this ParallelVehicle from others.
        capacity: int
            How many `Payload`s the vehicle can be loaded on this vehicle.
        current_rail_node_id: int
            The two RailNetwork nodes whose connecting edge the vehicle is currently in.
        rail_network: RailNetwork
            The RailNetwork that the vehicle will be moving within.
        destination: RailTerminal, optional
            The destination to give the vehicle, if any.
        current_payload: list[Payload] | Payload (optional)
            The Payload(s) to initialize the vehicle with, if any.
        """
        self.env = env
        self.current_rail_node_id = current_rail_node_id
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.current_destination = destination
        self._rail_network = rail_network
        self.current_payload: list[Payload] = []
        self.current_velocity_ms = 0  # start the vehicle stationary.
        if current_payload:
            if isinstance(current_payload, list):
                self.current_payload += current_payload
            elif isinstance(current_payload, Payload):
                self.current_payload.append(current_payload)
        self.reporting_list: list[dict[str, float]] = []
        self.current_route: list[int] = current_route or []
        self.current_terminal = self._rail_network.check_for_rail_terminal(
            current_rail_node_id
        )

    def __format__(self, spec):
        """
        Convenience method for pretty-printing ParallelVehicle state.
        """
        s = f"Parallel Vehicle {self.vehicle_id}\n"
        if self.current_terminal is not None:
            s += f"At rail terminal: {self.current_terminal.name}\n"
        s += f"At rail ID: {self.current_rail_node_id}"

        if self.current_payload:
            s += f"\nLoaded with {len(self.current_payload)}/{self.capacity} payloads"
        if self.current_destination:
            s += f"\nGoing to: {self.current_destination.name}"
        return s

    def run(self):
        """
        Executes the full vehicle logic loop.
        """
        self._report_current_state()
        while True:
            # If we have a route, move along it.
            if self.current_route:
                yield from self.traverse_along_route()
            else:
                # If we are in a terminal right now, execute terminal ops.
                if self.current_terminal is not None:
                    yield from self.current_terminal.process_incoming_rail_vehicle(self)

                # TODO: Add logic for assigning a new destination rail node id and make a route
                # Need to get from: Payload -> Assigned CY -> Nearest Rail Terminal -> Node ID.
                # Then we can get just query the RailNetwork for the route.
                # Should that be the job of Vehicle? RailNetwork?  Or a separate Router??
                if self.current_payload:
                    destination_rail_node_id = (
                        self._rail_network.get_nearest_rail_id_from_payload(
                            self.current_payload[0]
                        )
                    )
                else:
                    # TODO: Have some sort of failsafe behavior? Return to previous terminal?  Right
                    # now we just keep waiting.
                    destination_rail_node_id = self.current_rail_node_id

                # Compute a route from the new destination.
                self.current_route = self._rail_network.compute_route(
                    self.current_rail_node_id, destination_rail_node_id
                )

    def traverse_along_route(self):
        """
        Logic for traversing along the current route.  Pops the next rail_node_id from the route,
        computes how long it will take to travel this distance, delays for this time, then updates
        the current_rail_node_id.
        """
        # First, get the next node to travel to.
        next_node = self.current_route.pop(0)
        print(
            f"T = {float(self.env.now):.03f} | "
            + f"Parallel vehicle leaving {self.current_rail_node_id}, "
            + f"headed to {next_node}."
            + f"{len(self.current_payload)}/{self.capacity})"
        )

        # If there is still more track to traverse after this node, keep the current speed.
        if self.current_route:
            terminal_velocity = self.current_velocity_ms
        else:
            terminal_velocity = 0

        # Compute the time it will take to traverse this segment.
        distance_to_travel_m = self._rail_network.get_segment_length_m(
            self.current_rail_node_id, next_node
        )
        time_to_travel_hrs = self._compute_travel_time_hrs(
            distance_to_travel_m, self.current_velocity_ms, terminal_velocity
        )

        # Delay the appropriate time, updating the current_rail_node_id when done.
        self.current_rail_node_id = yield self.env.timeout(
            time_to_travel_hrs, value=self.current_rail_node_id
        )
        # Check if we arrived at a terminal at this node.
        self.current_terminal = self._rail_network.check_for_rail_terminal(
            self.current_rail_node_id
        )

    def _compute_travel_time_hrs(
        self,
        distance_to_travel_m: float,
        initial_velocity_ms: float,
        terminal_velocity_ms: float,
    ):
        """
        Helper function that computes how long it takes the vehicle to travel a given length of
        rail. Right now this just assumes constant acceleration such that the terminal velocity is
        hit at the end of the distance traveled.  In the future, should accomodate acceleration
        limits on the vehicle.

        Parameters:
        ----------
        distance_to_travel_m: float
            How far the vehicle will travel, in meters.
        initial_velocity_ms: float
            How fast the vehicle is moving at the beginning of the interval, in meters per second.
        terminal_velocity_ms: float
            How fast the vehicle should be moving at the end of the interval, in meters per second.

        Returns:
        --------
        How long it will take the vehicle to travel this distance, in hours.
        """
        return (
            2.0
            * distance_to_travel_m
            / (initial_velocity_ms + terminal_velocity_ms)
            / 60
            / 60
        )

    def drive_between_terminals(self):
        """
        DEPRECATED
        Wrapper function for behavior during travel between terminals.  Right
        now this is just a delay, but it could be more complicated if needed.
        """
        # Announce departure of the vehicle.
        print(
            f"T = {float(self.env.now):.03f} | "
            + f"Parallel vehicle leaving {self.current_terminal.name}"
            + f" with {len(self.current_payload)} payloads"
        )

        # Travel between the terminals. Right now this is just a delay.
        time_to_travel_hrs = self._time_to_travel_hrs()
        previous_location = yield self.env.timeout(
            time_to_travel_hrs, value=self.current_terminal
        )

        # Mark that the vehicle has arrived by setting its location.
        self.current_terminal = self.current_destination
        self._report_current_state()

        # Announce arrival of the vehicle.
        print(
            f"T = {float(self.env.now):.03f} | "
            + f"Parallel vehicle arrived at {self.current_terminal.name}, dropping off."
        )

        # Set the vehicle to turn back around.
        self.current_destination = previous_location

    def pickup_payloads(self, terminal: RailTerminal):
        """
        DEPRECATED
        Asks the vehicle to pick up new Payloads from the given RailTerminal until it is full.
        """
        while len(self.current_payload) < self.capacity:
            # while not at capacity, add Payloads to the vehicle.
            payload = yield from terminal.get_from_departure_queue()
            time_to_load_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(delay=time_to_load_hrs, value=payload)
            self.current_payload.append(payload)
            payload.set_location(self)
            self._report_current_state()
            print(
                f"T = {float(self.env.now):.03f} | "
                + f"Parallel Vehicle picked up a payload at {terminal.name} "
                + f"({len(self.current_payload)}/{self.capacity})"
            )

    def dropoff_payloads(self):
        """
        Asks the vehicle to drop off all of its current Payloads at its current RailTerminal.
        """
        while self.current_payload:
            # Removes payloads from vehicle in LIFO order.
            payload = self.current_payload.pop()

            # Set the current location of the payload to where the vehicle is.
            payload.set_location(self.current_terminal)

            # Delay some time for unloading.
            time_to_unload_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(delay=time_to_unload_hrs, value=payload)

            yield from self.current_terminal.put_in_arrival_queue(payload)
            self._report_current_state()
            print(
                f"T = {float(self.env.now):.03f} | "
                + f"Parallel Vehicle dropped off a payload at {self.current_terminal.name} "
                + f"({len(self.current_payload)}/{self.capacity})"
            )

    def current_payload_count(self) -> int:
        """
        Returns the current number of Payloads on the Parallel Vehicle.
        """
        return len(self.current_payload)

    def payload_capacity(self) -> int:
        return self.capacity

    def get_payload(self):
        """
        Gets the next Payload from the current_payload list.  Currently calls `pop()` so this is a
        LIFO operation.
        """
        if self.current_payload:
            time_to_load_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(
                delay=time_to_load_hrs, value=self.current_payload.pop()
            )
            self._report_current_state()
            return payload
        return None

    def put_payload(self, payload: Payload):
        """
        Puts a given Payload object onto the ParallelVehicle.  Currentlly calls `.append()` so it
        adds to the end of the list.
        """
        if len(self.current_payload) < self.capacity:
            time_to_load_hrs = self._time_to_load_hrs()
            # Delay for processing.
            payload = yield self.env.timeout(delay=time_to_load_hrs, value=payload)
            self.current_payload.append(payload)
            self._report_current_state()

    def _report_current_state(self):
        """
        Internal method that builds and appends the current state to the
        reporting_list.
        """
        self.reporting_list.append(self._get_timestamped_state())

    def _get_timestamped_state(self):
        """
        Internal method that builds the current state dictionary.
        """
        return {
            "TIME": self.env.now,
            "LATITUDE": self.current_terminal.get_location().get_latitude(),
            "LONGITUDE": self.current_terminal.get_location().get_longitude(),
            "CURRENT_LOAD": len(self.current_payload),
        }

    def _time_to_travel_hrs(self):
        """
        Method to report how long it takes the vehicle to travel between the RailTerminals.
        Right now, generates a random number, but in the future could be more complicated.
        """
        return random.uniform(5, 15)

    def _time_to_load_hrs(self):
        """
        Method to report how long it takes to load a Payload onto the vehicle.
        Right now, generates a random number, but in the future could be more
        complicated.
        """
        # TODO: (@ps-jbunton) This should be an attribute or function of
        # vehicle/payload in question, not hard-coded
        return random.expovariate(0.25)  # Rate parameter means 0.25 hours avg load time


class Truck(PayloadLocation):
    """
    Truck class.  Models trucks that can pick up and drop off containers from `ContainerYard`s and
    `RailTerminal`s, while traveling by road.

    Attributes
    ----------
    env: simpy.Environment
        The simpy environment the Truck will live in.
    vehicle_id: int
        A unique int to help distinguish this `Truck` from others.
    capacity: int
        An int characterizing how many `Payload`s the truck can carry at once.
    current_payload: list[Payload]
        A list of the `Payload`s currently loaded onto this truck, or None if there aren't any.
    current_location: Terminal | ContainerYard
        The current location of the truck.
    current_destination: Terminal | ContainerYard | None
        Where the truck is headed next, or None if it is waiting.
    reporting_list: list[dict[str, value]]
        A list of reported statuses or events, where the elements are dicts
        keyed by strings representing the quantity and values representing the
        reported status/quantity.
    """

    def __init__(
        self,
        env: simpy.Environment,
        vehicle_id: int,
        capacity: int,
        current_location: RailTerminal | ContainerYard,
        current_payload: list[Payload] | Payload | None = None,
        current_destination: RailTerminal | ContainerYard | None = None,
    ) -> None:
        self.env = env
        self.vehicle_id = vehicle_id
        self.current_location = current_location
        self.current_destination = current_destination
        self.capacity = capacity
        self.current_payload = []
        if current_payload:
            if isinstance(current_payload, list):
                self.current_payload += current_payload
            elif isinstance(current_payload, Payload):
                self.current_payload.append(current_payload)
        self.reporting_list: list[dict[str, float]] = []

    def __format__(self, spec) -> str:
        """
        Convenience method for printing out status of Truck.
        """
        s = f"Truck {self.vehicle_id}\nAt location: {self.current_location.name}\n"
        if self.current_payload:
            s += f"Loaded with {len(self.current_payload)} payloads \n"
        else:
            s += "Empty\n"
        if self.current_destination:
            s += f"Going to: {self.current_destination.name}"
        return s

    def run(self):
        """
        Runs the Truck logic.
        """
        while True:
            # Pickup any payloads from the current location if it has an arrival queue.
            if hasattr(self.current_location, "arrival_queue"):
                yield from self.pickup_payloads()

            # Drive to the current destination (assumes we have one!)
            print(
                f"T = {float(self.env.now):.03f} | "
                + f"Truck {self.vehicle_id} departing {self.current_location.name} "
                + f"for {self.current_destination.name}"
            )
            yield from self.drive_to_destination()

            # Drop off payloads at current location (will only execute if we have payloads).
            if self.current_payload:
                yield from self.dropoff_payloads()

    def drive_to_destination(self):
        """
        Method that characterizes behavior during the driving sequence.  Right now it's just a
        randomized delay,  but in the future it could do something more complicated.
        """
        time_to_destination_hrs = self._time_to_destination_hrs()
        yield self.env.timeout(
            delay=time_to_destination_hrs, value=self.current_location
        )

        # Hold on to the previous location.
        previous_location = self.current_location
        # Mark that the vehicle has arrived.
        self.current_location = self.current_destination
        print(
            f"T = {float(self.env.now):.03f} | Truck {self.vehicle_id}"
            + f" arrived at {self.current_location.name}"
        )
        # Log this in the report.
        self._report_current_state()
        # Tell the truck to turn around after dropping its payloads (may be overwritten if we
        # pick up new payloads).
        self.current_destination = previous_location

    def dropoff_payloads(self):
        """
        Method describing behavior during the dropoff sequence at a ContainerYard.  Right now it
        just dumps its contents into the `ContainerYard`s `payload_sink`.
        """
        while self.current_payload:
            # Pull and deliver the payloads in LIFO order
            payload = self.current_payload.pop()

            # Add some delay for payload processing
            time_to_dropoff_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(delay=time_to_dropoff_hrs, value=payload)

            payload.set_location(self.current_location)
            yield from self.current_location.deliver_order(payload)

            # Log that we just dropped off a payload.
            self._report_current_state()
            print(
                f"T = {float(self.env.now):.03f} | Truck {self.vehicle_id}"
                + f" dropped off a payload at {self.current_location.name}"
                + f" ({len(self.current_payload)}/{self.capacity})"
            )

    def pickup_payloads(self):
        """
        Method describing behavior during the pickup sequence at a RailTerminal.  It starts by
        grabbing the first available container in the `RailTerminal`s `arrival_queue`.  If the
        `Truck`'s capacity is not met and there is another container in the
        `RailTerminal`'s `arrival_queue` with the same `destination` field, it pulls it.
        """
        # Pickup the first available payload.
        if len(self.current_payload) < self.capacity:
            payload = yield from self.current_location.get_from_arrival_queue()

            time_to_load_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(delay=time_to_load_hrs, value=payload)
            payload.set_location(self)
            self.current_destination = payload.destination
            self.current_payload.append(payload)

            # Log that we just picked up a payload.
            self._report_current_state()
            print(
                f"T = {float(self.env.now):.03f} | Truck {self.vehicle_id} "
                + f"picked up payload destined for {payload.destination.name} "
                + f"from {self.current_location.name} ({len(self.current_payload)}/{self.capacity})"
            )

        # Expression to filter out payloads with the same destination.
        def destination_filter(payload):
            return payload.destination == self.current_destination

        while (
            len(self.current_payload) < self.capacity
            and sum(
                destination_filter(payload)
                for payload in self.current_location.arrival_queue.items
            )
            > 0
        ):
            payload = yield from self.current_location.get_from_arrival_queue(
                destination_filter
            )
            time_to_load_hrs = self._time_to_load_hrs()
            payload = yield self.env.timeout(delay=time_to_load_hrs, value=payload)
            self.current_payload.append(payload)

            # Log that we picked up another payload.
            self._report_current_state()
            print(
                f"T = {float(self.env.now):.03f} | Truck {self.vehicle_id} "
                + f"picked up a payload destined for {payload.destination.name} from "
                + f"{self.current_location.name} ({len(self.current_payload)}/{self.capacity})"
            )

    def _time_to_destination_hrs(self):
        """
        Function that returns how long it takes to travel from the self.current_location to
        self.current_destination.  Right now, it's just a random number, but we could do something
        more complicated.
        """
        return random.uniform(0.5, 3.0)

    def _time_to_load_hrs(self):
        """
        Method to report how long it takes to load a Payload onto the vehicle.
        Right now, generates a random number, but in the future could be more
        complicated.
        """
        # TODO: (@ps-jbunton) This should be an attribute or function of
        # vehicle/payload in question, not hard-coded
        return random.expovariate(0.25)  # Rate parameter means 0.25 hours avg load time

    def _report_current_state(self):
        """
        Internal method for building and appending the current state dict.
        """
        self.reporting_list.append(self._get_timestamped_state())

    def _get_timestamped_state(self):
        """
        Internal method for building the current state dict.
        """
        return {
            "TIME": self.env.now,
            "LATITUDE": self.current_location.location.get_latitude(),
            "LONGITUDE": self.current_location.location.get_longitude(),
            "CURRENT_LOAD": len(self.current_payload),
        }
