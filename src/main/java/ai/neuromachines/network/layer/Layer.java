package ai.neuromachines.network.layer;

public interface Layer {

    int nodeCount();

    /**
     * Current input signals
     */
    float[] input();

    /**
     * Current output signals
     */
    float[] output();

    /**
     * Returns copy of this layer.
     * Weights are copied. Input and output signals are not.
     * Previous layer is set to provided by argument.
     *
     * @return the deep copy object of the same class as for this object
     * @apiNote Implementation must call constructor to do copy
     */
    Layer copyOf(Layer previous);
}
