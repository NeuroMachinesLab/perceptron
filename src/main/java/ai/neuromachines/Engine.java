package ai.neuromachines;

import ai.neuromachines.network.ResponseLayer;
import ai.neuromachines.network.SensorLayer;
import ai.neuromachines.network.function.ActivationFunc;

import java.time.Duration;
import java.time.Instant;

public class Engine {
    public static void main(String[] args) {

        Instant t0 = Instant.now();
        int outputNodeCnt = 2;
        float[] expected = new float[]{0.1f, 0.9f};
        print("Expected", expected);

        ActivationFunc func = ActivationFunc.sigmoid(1);
        float[] input = new float[3];
        random(input);
        print("Inputs", input);

        SensorLayer sensorLayer = new SensorLayer(input);
        ResponseLayer layer1 = new ResponseLayer(4, sensorLayer, func);
        ResponseLayer layer2 = new ResponseLayer(outputNodeCnt, layer1, func);

        for (int i = 0; i < 2_000; i++) {
            System.out.print("Iteration #");
            System.out.println(i + 1);
            print("Layer 1 Weight", layer1.weights());
            print("Layer 2 Weight", layer2.weights());
            print("Outputs", layer2.output());
            layer2.correctWeights(expected);
        }
        print("Expected", expected);

        System.out.print("Time spent: ");
        System.out.println(Duration.between(t0, Instant.now()));
    }

    private static void random(float[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
    }

    private static void print(String message, float[] a) {
        System.out.println(message + ":");
        for (float v : a) {
            System.out.println(v);
        }
        System.out.println();
    }

    private static void print(String message, float[][] a) {
        System.out.println(message + ":");
        for (float[] innerA : a) {
            for (float v : innerA) {
                System.out.print(v);
                System.out.print("\t");
            }
            System.out.println();
        }
        System.out.println();
    }
}
