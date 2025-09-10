package ai.neuromachines.network.layer;

import ai.neuromachines.Assert;
import lombok.AllArgsConstructor;

import static lombok.AccessLevel.PRIVATE;

/**
 * Layer with no input signals. Produces only output signal
 */
@AllArgsConstructor(access = PRIVATE)
public class SensorLayer implements Layer {
    private float[] signal;

    public static SensorLayer of(int nodeCnt) {
        return new SensorLayer(new float[nodeCnt]);
    }

    public void setOutput(float[] signal) {
        Assert.isTrue(signal.length == this.signal.length, "Incorrect array length");
        this.signal = signal;
    }

    @Override
    public int nodeCount() {
        return signal.length;
    }

    @Override
    public float[] output() {
        return signal;
    }
}
