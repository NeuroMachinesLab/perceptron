package ai.neuromachines.network.layer;


/**
 * Layer produces only output signal and doesn't have activation function.
 */
public interface SensorLayer extends Layer {

    static SensorLayer of(int nodeCnt) {
        return SensorLayerImpl.of(nodeCnt);
    }

    void setInput(float[] signal);
}
