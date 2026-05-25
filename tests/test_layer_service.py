import pytest

from neural_network import LayerService, PerceptronService


def test_predict_returns_output_for_each_perceptron():
    layer = LayerService(
        [
            PerceptronService.from_weights([1, 0], threshold=1),
            PerceptronService.from_weights([0, 1], threshold=1),
        ]
    )

    assert layer.predict([1, 0]) == [1.0, 0.0]


def test_copies_perceptrons_into_internal_list():
    perceptrons = [PerceptronService.from_weights([1])]
    layer = LayerService(perceptrons)
    perceptrons.append(PerceptronService.from_weights([0]))

    assert len(layer.perceptrons) == 1


def test_rejects_non_perceptron_values():
    with pytest.raises(TypeError):
        LayerService([object()])
