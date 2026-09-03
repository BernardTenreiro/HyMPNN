"""Reusable equivariant message-passing layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def segment_sum(data: Tensor, segment_ids: Tensor, segment_count: int) -> Tensor:
    """Sum rows of ``data`` according to integer segment identifiers."""
    result = data.new_zeros((segment_count, data.size(1)))
    expanded_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, expanded_ids, data)
    return result


class EquivariantGraphConvolution(nn.Module):
    """Core E(n)-equivariant graph convolution used by the dense model."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int,
        edge_features: int = 0,
        node_attribute_features: int = 0,
        activation: nn.Module | None = None,
        recurrent: bool = True,
        coordinate_weight: float = 1.0,
        attention: bool = False,
        normalize_differences: bool = False,
        bounded_coordinates: bool = False,
    ) -> None:
        super().__init__()
        activation = activation or nn.ReLU()
        self.coordinate_weight = coordinate_weight
        self.recurrent = recurrent
        self.attention = attention
        self.normalize_differences = normalize_differences

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_features * 2 + 1 + edge_features, hidden_features),
            activation,
            nn.Linear(hidden_features, hidden_features),
            activation,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_features + input_features + node_attribute_features, hidden_features),
            activation,
            nn.Linear(hidden_features, output_features),
        )

        coordinate_output = nn.Linear(hidden_features, 1, bias=False)
        nn.init.xavier_uniform_(coordinate_output.weight, gain=0.001)
        coordinate_layers: list[nn.Module] = [
            nn.Linear(hidden_features, hidden_features),
            activation,
            coordinate_output,
        ]
        if bounded_coordinates:
            coordinate_layers.append(nn.Tanh())
            self.coordinate_range = nn.Parameter(torch.ones(1) * 3)
        self.coord_mlp = nn.Sequential(*coordinate_layers)

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_features, 1), nn.Sigmoid())

    def edge_model(
        self,
        source: Tensor,
        target: Tensor,
        radial: Tensor,
        edge_attributes: Tensor | None,
    ) -> Tensor:
        inputs = [source, target, radial]
        if edge_attributes is not None:
            inputs.append(edge_attributes)
        messages = self.edge_mlp(torch.cat(inputs, dim=1))
        if self.attention:
            messages = messages * self.att_mlp(messages)
        return messages

    def node_model(
        self,
        nodes: Tensor,
        edge_index: tuple[Tensor, Tensor] | list[Tensor],
        edge_attributes: Tensor,
        node_attributes: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        rows, _ = edge_index
        aggregate = segment_sum(edge_attributes, rows, nodes.size(0))
        inputs = [nodes, aggregate]
        if node_attributes is not None:
            inputs.append(node_attributes)
        output = self.node_mlp(torch.cat(inputs, dim=1))
        if self.recurrent:
            output = nodes + output
        return output, aggregate

    def coordinate_model(
        self,
        coordinates: Tensor,
        edge_index: tuple[Tensor, Tensor] | list[Tensor],
        coordinate_differences: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        rows, _ = edge_index
        translations = coordinate_differences * self.coord_mlp(edge_features)
        translations = translations.clamp(min=-100, max=100)
        counts = coordinates.new_zeros((coordinates.size(0), translations.size(1)))
        aggregate = coordinates.new_zeros((coordinates.size(0), translations.size(1)))
        expanded_rows = rows.unsqueeze(-1).expand_as(translations)
        aggregate.scatter_add_(0, expanded_rows, translations)
        counts.scatter_add_(0, expanded_rows, torch.ones_like(translations))
        return coordinates + aggregate / counts.clamp(min=1) * self.coordinate_weight

    def coordinate_features(
        self,
        edge_index: tuple[Tensor, Tensor] | list[Tensor],
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rows, columns = edge_index
        differences = coordinates[rows] - coordinates[columns]
        radial = differences.square().sum(dim=1, keepdim=True)
        if self.normalize_differences:
            differences = differences / (radial.sqrt() + 1)
        return radial, differences

    def forward(
        self,
        nodes: Tensor,
        edge_index: tuple[Tensor, Tensor] | list[Tensor],
        coordinates: Tensor,
        edge_attributes: Tensor | None = None,
        node_attributes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        rows, columns = edge_index
        radial, differences = self.coordinate_features(edge_index, coordinates)
        messages = self.edge_model(nodes[rows], nodes[columns], radial, edge_attributes)
        coordinates = self.coordinate_model(coordinates, edge_index, differences, messages)
        nodes, _ = self.node_model(nodes, edge_index, messages, node_attributes)
        return nodes, coordinates, edge_attributes
