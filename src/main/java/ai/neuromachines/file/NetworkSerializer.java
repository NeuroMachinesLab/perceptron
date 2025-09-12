package ai.neuromachines.file;

import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.function.SigmoidFunc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

public class NetworkSerializer {

    private static final SigmoidFunc SIGMOID = ActivationFunc.sigmoid(1);

    // TODO write activation function
    /**
     * Writes network to specified byte channel
     */
    public static void serialize(Network network, WritableByteChannel out) throws IOException {
        for (int layer = 0, cnt = network.layersCount() - 1; layer < cnt; layer++) {
            float[][] weights = network.weights(layer);
            ByteBuffer buffer = printMatrixToUtf8String(weights);
            while (buffer.hasRemaining()) {
                out.write(buffer);
            }
        }
    }

    private static ByteBuffer printMatrixToUtf8String(float[][] weights) {
        StringBuilder sb = new StringBuilder(weights.length * weights[0].length * 10);
        for (float[] row : weights) {
            for (float v : row) {
                sb.append(v).append(" ");
            }
            sb.append("\n");  // end of line
        }
        sb.append("\n");  // end of matrix
        byte[] bytes = sb.toString()
                .getBytes(UTF_8);
        return ByteBuffer.wrap(bytes);
    }

    /**
     * Creates new Network with parameters provided by {@code ch}
     */
    public static Network deserialize(ReadableByteChannel ch) throws IOException {
        InputStream is = Channels.newInputStream(ch);
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
