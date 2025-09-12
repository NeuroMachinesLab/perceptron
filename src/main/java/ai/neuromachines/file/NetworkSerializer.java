package ai.neuromachines.file;

import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.function.SigmoidFunc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

public class NetworkSerializer {

    private static final SigmoidFunc SIGMOID = ActivationFunc.sigmoid(1);

    /**
     * @return ReadableByteChannel with network parameters
     */
    public static ReadableByteChannel serialize(Network network) {
        return NetworkReadableByteChannel.of(network);
    }

    /**
     * Creates new Network with parameters provided by {@code from}
     */
    public static Network deserialize(ReadableByteChannel from) throws IOException {
        InputStream is = Channels.newInputStream(from);
        InputStreamReader isr = new InputStreamReader(is, UTF_8);
        BufferedReader br = new BufferedReader(isr);
        List<float[][]> layersWeights = new ArrayList<>();
        float[][] weights;
        while ((weights = readMatrix(br)).length > 0) {
            layersWeights.add(weights);
        }
        float[][][] networkWeights = convertToArray3(layersWeights);
        return Network.of(SIGMOID, networkWeights);
    }

    private static float[][] readMatrix(BufferedReader br) throws IOException {
        String line;
        List<float[]> weights = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            if (line.isEmpty()) {
                break; // end of matrix
            }
            float[] row = parseRow(line);
            weights.add(row);
        }
        return convertToArray2(weights);
    }

    private static float[] parseRow(String line) {
        String[] values = line.split(" ");
        float[] row = new float[values.length];
        int i = 0;
        for (String v : values) {
            row[i++] = Float.parseFloat(v);
        }
        return row;
    }

    private static float[][] convertToArray2(List<float[]> list) {
        float[][] result = new float[list.size()][];
        int i = 0;
        for (float[] row : list) {
            result[i++] = row;
        }
        return result;
    }

    private static float[][][] convertToArray3(List<float[][]> list) {
        float[][][] result = new float[list.size()][][];
        int i = 0;
        for (float[][] row : list) {
            result[i++] = row;
        }
        return result;
    }
}
