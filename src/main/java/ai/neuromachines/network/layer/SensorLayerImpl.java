package ai.neuromachines.network.layer;

import ai.neuromachines.Assert;
import lombok.AllArgsConstructor;

import static lombok.AccessLevel.PRIVATE;

@AllArgsConstructor(access = PRIVATE)
class SensorLayerImpl implements Layer, SensorLayer {
    private float[] signal;

    static SensorLayer of(int nodeCnt) {
        return new SensorLayerImpl(new float[nodeCnt]);
    }

    @Override
    public void setInput(float[] signal) {
        Assert.isTrue(signal.length == this.signal.length, "Incorrect array length");
        this.signal = signal;
    }

    @Override
    public int nodeCount() {
        return signal.length;
    }

    @Override
    public float[] input() {
        return signal;
    }

    @Override
    public float[] output() {
        return signal;
    }
}
