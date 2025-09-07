package ai.neuromachines;

import ai.neuromachines.network.ResponseLayer;
import ai.neuromachines.network.SensorLayer;
import ai.neuromachines.network.function.ActivationFunc;

public class Engine {
    public static void main(String[] args) {
        ActivationFunc func = ActivationFunc.sigmoid(1);
        float[] input = new float[2];
        random(input);
        print("Inputs", input);

        SensorLayer sensorLayer = new SensorLayer(input);
        ResponseLayer layer1 = new ResponseLayer(4, sensorLayer, func);
        ResponseLayer layer2 = new ResponseLayer(3, layer1, func);

        float[] output = layer2.output();
        print("Outputs", output);
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
}
