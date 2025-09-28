package ai.neuromachines.examples;

import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.train.TrainStrategy;

import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

import static java.nio.file.StandardOpenOption.*;

/**
 * First run creates Perceptron with 3 layers (sensor, hidden and output).
 * Network weights set to random values.
 * For hidden layer activation function set to Leaky ReLu, for output layer - set to Softmax .
 * <p>
 * Network input layer consists of 3 nodes. Network input signal set to 0.1, 0.2, 0.3 for each of those nodes.
 * Network output layer consists of 2 nodes. Expected output values is 0.2 and 0.8.
 * Network is trained in 100 cycles by Backpropagation algorithm.
 * Corrected network weights is saved to "network.txt" file.
 * <p>
 * Second and other runs read network weights form "network.txt" file.
 * Network is trained again in 100 cycles.
 * Corrected network weights rewrites to "network.txt" file.
 * <p>
 * "network.txt" file helps to split work to sequential parts. If training is takes a lot of time,
 * this pattern helps to take the network snapshot at some middle point and continue training late from this snapshot.
 * <p>
 * If you need to start network learning from scratch, delete the "network.txt" file.
 */
public class TrainingSample {

    private static final Path path = Path.of("network.txt");


    public static void main(String[] args) throws IOException {

        float[] input = new float[]{0.1f, 0.2f, 0.3f};
        float[] expectedOutput = new float[]{0.2f, 0.8f};

        Network network = Files.exists(path) ?
                openNetworkFromFile(path) :
                createNetwork(input.length, 4, expectedOutput.length);

        TrainStrategy trainStrategy = TrainStrategy.backpropagation(network);
        trainNetwork(input, expectedOutput, trainStrategy, 100);
        printResult(input, expectedOutput, network);

        saveToFile(network, path);
    }

    private static Network createNetwork(int... layersNodeCount) {
        System.out.println("Create network with random weights and " +
                layersNodeCount[0] + " nodes in input layer, " +
                layersNodeCount[1] + " nodes in hidden layer, " +
                layersNodeCount[2] + " nodes in output layer");
        ActivationFunc hiddenLayerActFunc = ActivationFunc.leakyReLu(0.01f);
        ActivationFunc outputLayerActFunc = ActivationFunc.softmax();
        return Network.of(List.of(hiddenLayerActFunc, outputLayerActFunc), layersNodeCount);
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

    private static void trainNetwork(float[] input,
                                     float[] expectedOutput,
                                     TrainStrategy trainStrategy,
                                     @SuppressWarnings("SameParameterValue") int iterations) {
        Instant t0 = Instant.now();
        for (int i = 0; i < iterations; i++) {
            trainStrategy.train(input, expectedOutput);
//            System.out.print("Iteration #");
//            System.out.println(i + 1);
//            print("Network", network);  // print weights
//            print("Outputs", network.output());
        }
        Duration timeSpent = Duration.between(t0, Instant.now());
        System.out.println("Train iterations: " + iterations);
        System.out.println("Train time: " + timeSpent);
    }

    private static void printResult(float[] input, float[] expectedOutput, Network network) throws IOException {
        System.out.println("--- Result ---");
        print("Inputs", input);
        print("Expected output", expectedOutput);
        print("Outputs", network.output());
        print("Network", network);
    }

    private static void print(String message, float[] a) {
        System.out.println(message + ":");
        for (float v : a) {
            System.out.println(v);
        }
        System.out.println();
    }

    private static void print(@SuppressWarnings("SameParameterValue") String message,
                              Network network) throws IOException {
        System.out.println(message + ":");
        WritableByteChannel ch = Channels.newChannel(System.out);
        NetworkSerializer.serialize(network, ch);
    }
}
