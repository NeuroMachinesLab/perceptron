package ai.neuromachines;

import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;

import static java.nio.file.StandardOpenOption.*;

public class Engine {
    private static final Path path = Path.of("network.txt");


    public static void main(String[] args) throws IOException {

        float[] input = new float[]{0.1f, 0.2f, 0.3f};
        float[] expectedOutput = new float[]{0.8f, 0.9f};

        Network network = Files.isRegularFile(path) ?
                openNetworkFromFile(path) :
                createNetwork(input, expectedOutput);

        trainNetwork(network, expectedOutput, 100);
        printResult(input, expectedOutput, network);

        saveToFile(network, path);
    }

    private static Network createNetwork(float[] input, float[] expectedOutput) {
        System.out.println("Create network with random weights");
        ActivationFunc func = ActivationFunc.sigmoid(1);
        Network network = Network.of(func, input.length, 4, expectedOutput.length);
        network.input(input);
        return network;
    }

    private static Network openNetworkFromFile(@SuppressWarnings("SameParameterValue") Path path) throws IOException {
        System.out.println("Read network from: " + path);
        try (FileChannel ch = FileChannel.open(path)) {
            return NetworkSerializer.deserialize(ch);
        }
    }

    private static void saveToFile(Network network,
                                   @SuppressWarnings("SameParameterValue") Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, CREATE, WRITE, TRUNCATE_EXISTING)) {
            NetworkSerializer.serialize(network, ch);
        }
        System.out.println("Network has been written to: " + path);
    }

    private static void trainNetwork(Network network,
                                     float[] expectedOutput,
                                     @SuppressWarnings("SameParameterValue") int iterations) {
        Instant t0 = Instant.now();
        network.output();  // calculate first result
        for (int i = 1; i < iterations; i++) {
//            System.out.print("Iteration #");
//            System.out.println(i + 1);
//            print("Layer 1 edges weights", network.transposedWeights(1));
//            print("Layer 2 edges weights", network.transposedWeights(2));
//            print("Outputs", network.output());
            network.train(expectedOutput);
        }
        Duration timeSpent = Duration.between(t0, Instant.now());
        System.out.println("Train iterations: " + iterations);
        System.out.println("Train time: " + timeSpent);
    }

    private static void printResult(float[] input, float[] expectedOutput, Network network) {
        System.out.println("--- Result ---");
        print("Inputs", input);
        print("Expected output", expectedOutput);
        print("Outputs", network.output());
        print("Layer 0-1 weights", network.weights(0));
        print("Layer 1-2 weights", network.weights(1));
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
