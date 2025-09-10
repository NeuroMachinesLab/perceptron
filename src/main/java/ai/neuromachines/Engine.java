package ai.neuromachines;

import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;

import java.time.Duration;
import java.time.Instant;

public class Engine {
    public static void main(String[] args) {

        Instant t0 = Instant.now();

        float[] input = new float[]{0.1f, 0.2f, 0.3f};
        float[] expectedOutput = new float[]{0.8f, 0.9f};

        Network network = createNetwork(input, expectedOutput);

        trainNetwork(network, expectedOutput, 2_000);
        printResult(input, expectedOutput, network);

        Duration timeSpent = Duration.between(t0, Instant.now());
        System.out.println("Time spent: " + timeSpent);
    }

    private static Network createNetwork(float[] input, float[] expectedOutput) {
        ActivationFunc func = ActivationFunc.sigmoid(1);
        Network network = Network.of(func, input.length, 4, expectedOutput.length);
        network.input(input);
        return network;
    }

    private static void trainNetwork(Network network,
                                     float[] expectedOutput,
                                     @SuppressWarnings("SameParameterValue") int iterations) {
        for (int i = 0; i < iterations; i++) {
            System.out.print("Iteration #");
            System.out.println(i + 1);
            print("Layer 1 edges weights", network.weights(1));
            print("Layer 2 edges weights", network.weights(2));
            print("Outputs", network.output());
            network.train(expectedOutput);
        }
    }

    private static void printResult(float[] input, float[] expectedOutput, Network network) {
        System.out.println("--- Result ---");
        print("Inputs", input);
        print("Expected output", expectedOutput);
        print("Outputs", network.output());
        print("Layer 1 edges weights", network.weights(1));
        print("Layer 2 edges weights", network.weights(2));
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
