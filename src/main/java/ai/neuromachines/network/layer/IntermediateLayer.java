package ai.neuromachines.network.layer;

import ai.neuromachines.network.function.ActivationFunc;

public interface IntermediateLayer extends Layer {

    ActivationFunc activationFunc();

    /**
     * Returns weights for connections between this layer and previous layer nodes
     * (row count equals to this layer node count, cols count equals to previous layer node count)
     */
    float[][] weights();
}
