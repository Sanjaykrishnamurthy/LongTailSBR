import math
from _operator import itemgetter


class RecentNeighbor:
    def __init__(self, session_id, session, session_timestamp, sample_size=0, k=500, factor1=True, l1=3.54,
                 factor2=True, l2=20 * 24 * 3600, factor3=True, l3=3.54):
        self.k = k
        self.sample_size = sample_size
        self.session_all = session
        self.session_id_all = session_id
        self.session_timestamp_all = session_timestamp
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        # cache
        self.session_timestamp_cache = {}  
        self.item_session_cache = {} 
        self.session_item_cache = {}  
        self.session_item_list_cache = {}  
        for i, session in enumerate(self.session_all):
            sid = self.session_id_all[i]  
            self.session_timestamp_cache.update({sid: self.session_timestamp_all[i]})
            for item in session:
                session_map = self.item_session_cache.get(item)
                if session_map is None:
                    session_map = set()
                    self.item_session_cache.update({item: session_map})
                session_map.add(sid)

                item_map = self.session_item_cache.get(sid)
                if item_map is None:
                    item_map = set()
                    self.session_item_cache.update({sid: item_map})
                item_map.add(item)

                item_list = self.session_item_list_cache.get(sid)
                if item_list is None:
                    item_list = []
                    self.session_item_list_cache.update({sid: item_list})
                item_list += [item]

        self.current_session_weight_cache = {}
        self.current_timestamp = 0

    def find_neighbours(self, session_items, input_item):
        neighbour_session_items = []
        zero_list = [0] * 10
        neighbour_sessions = self.possible_neighbour_sessions(session_items, input_item)
        neighbour_sessions = list(set(list(neighbour_sessions)[:10]))
        zero_list[:len(neighbour_sessions)] = neighbour_sessions
        return zero_list
        # # get top k according to similarity
        # for session in neighbour_sessions:
        #     neighbour_session_items.append(list(self.session_item_cache.get(session)))
        # return neighbour_sessions

    def possible_neighbour_sessions(self, session_items, input_item):
        neighbours = set()
        for item in session_items:
            if item in self.item_session_cache:
                neighbours = neighbours | self.item_session_cache.get(item)
        return list(neighbours)

    def predict(self, session_id, session_items):
        last_item_id = session_items[-1]
        neighbour_session_items = self.find_neighbours(set(session_items), last_item_id)
        return neighbour_session_items
