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
}
