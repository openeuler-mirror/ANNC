from typing import List
from abc import ABC, abstractmethod
from .graph import Graph, Node


class CheckFailed(Exception):
    pass


class BaseRewriter(ABC):

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

        self.fused_ops: List[Node] = []

        for node in self.graph.nodes:
            try:
                self.match_and_rewrite(node)
            except CheckFailed:
                pass

    @abstractmethod
    def match_and_rewrite(self, node: Node):
        pass

    def check_node(self, node: Node, attr: tuple):
        """
        Verify whether the type and name of the node conforms to expectations.
        :param node: node object
        :param users: expected node type and name
        """
        type, name = attr
        if type is not None and node.type != type:
            raise CheckFailed
        if name is not None and node.name != name:
            raise CheckFailed

    def check_users(self,
                    node: Node,
                    users: List[tuple],
                    user_num: int = None):
        """
        Verify whether the list of users for the node conforms to expectations.
        :param node: node object
        :param users: expected list of user types and names
        :param expected_num_users: expected number of users (optional)
        """
        if self.graph.check_node_exist(node):
            if user_num is not None and len(node.users) != user_num:
                raise CheckFailed
            if len(users) > len(node.users):
                raise CheckFailed
            for i, (type, name) in enumerate(users):
                if type is not None and node.users[i].type != type:
                    raise CheckFailed
                if name is not None and node.users[i].name != name:
                    raise CheckFailed
        else:
            raise CheckFailed

    def check_operands(self,
                       node: Node,
                       operands: List[tuple],
                       operand_num: int = None):
        """
        Verify whether the list of operands for the node conforms to expectations.
        :param node: node object
        :param operands: expected list of operand types and names
        :param expected_operand_num: expected number of operands (optional)
        """
        if operand_num is not None and len(node.operands) != operand_num:
            raise CheckFailed
        if len(operands) > len(node.operands):
            raise CheckFailed
        for i, (type, name) in enumerate(operands):
            if type is not None and node.operands[i][0].type != type:
                raise CheckFailed
            if name is not None and node.operands[i][0].name != name:
                raise CheckFailed

    def replace_all_users_with(self, old_node: Node, old_index: int,
                               new_node: Node, new_index: int):
        for user in old_node.users:
            for i, operand in enumerate(user.operands):
                if operand[0].name == old_node.name and \
                    operand[1] == old_index:
                    user.operands[i] = (new_node, new_index)