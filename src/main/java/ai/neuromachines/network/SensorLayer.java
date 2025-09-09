package ai.neuromachines.network;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class SensorLayer implements Layer {
    private final float[] input;

    @Override
    public int nodeCount() {
        return input.length;
    }

    @Override
    public float[] output() {
        return input;
    }
}
