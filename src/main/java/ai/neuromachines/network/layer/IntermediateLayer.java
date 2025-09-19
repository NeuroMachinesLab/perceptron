package ai.neuromachines.network.layer;

import ai.neuromachines.network.function.ActivationFunc;

public interface IntermediateLayer extends Layer {

    ActivationFunc activationFunc();

    /**
     * Returns weights for connections between this layer and previous layer nodes
     * (row count equals to this layer node count, cols count equals to previous layer node count)
     */
    float[][] weights();

    /**
     * Train current and all previous layers weights, till first (sensor) layer
     *
     * @see <a href="https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8">Wiki</a>
     */
    void train(float[] expectedOutput);
}
