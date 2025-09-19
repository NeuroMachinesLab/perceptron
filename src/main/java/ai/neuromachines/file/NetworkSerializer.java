package ai.neuromachines.file;

import ai.neuromachines.Assert;
import ai.neuromachines.math.Matrix;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.layer.IntermediateLayer;
import ai.neuromachines.network.layer.Layer;
import ai.neuromachines.network.layer.ResponseLayer;
import ai.neuromachines.network.layer.SensorLayer;

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
import java.util.Objects;
import java.util.Optional;

import static java.nio.charset.StandardCharsets.UTF_8;

public class NetworkSerializer {

    private static final String SENSOR_LAYER_ACTIVATION_FUNC_SERIALIZED = "0: Identity(alpha=1.0)";

    /**
     * Writes network to specified byte channel
     */
    public static void serialize(Network network, WritableByteChannel out) throws IOException {
        ByteBuffer buffer = printSensorLayerActivationFunc();
        write(buffer, out);
        int layerIndex = 1;
        for (IntermediateLayer layer : network.intermediateLayers()) {
            ActivationFunc func = layer.activationFunc();
            float[][] weights = Matrix.transpose(layer.weights());
            buffer = printMatrixToUtf8String(layerIndex++, func, weights);
            write(buffer, out);
        }
    }

    private static ByteBuffer printSensorLayerActivationFunc() {
        byte[] bytes = (SENSOR_LAYER_ACTIVATION_FUNC_SERIALIZED + "\n").getBytes(UTF_8);
        return ByteBuffer.wrap(bytes);
    }

    private static ByteBuffer printMatrixToUtf8String(int layerIndex, ActivationFunc activationFunc, float[][] weights) {
        StringBuilder sb = new StringBuilder(weights.length * weights[0].length * 10);
        // weights between i-th and (i+1) layer
        for (float[] row : weights) {
            for (float v : row) {
                sb.append(v).append(" ");
            }
            sb.append('\n');  // end of line
        }
        sb.append('\n');  // end of matrix
        sb.append(layerIndex)
                .append(": ")
                .append(ActivationFuncSerializer.serialize(activationFunc))  // activation function for (i+1) layer
                .append('\n');
        byte[] bytes = sb.toString()
                .getBytes(UTF_8);
        return ByteBuffer.wrap(bytes);
    }

    private static void write(ByteBuffer buffer, WritableByteChannel out) throws IOException {
        while (buffer.hasRemaining()) {
            out.write(buffer);
        }
    }

    /**
     * Creates new Network with parameters provided by {@code ch}
     */
    public static Network deserialize(ReadableByteChannel ch) throws IOException {
        BufferedReader br = getBufferedReader(ch);
        String sensorLayerActFunc = br.readLine();
        Assert.isTrue(Objects.equals(sensorLayerActFunc, SENSOR_LAYER_ACTIVATION_FUNC_SERIALIZED), "Sensor Layer expected");
        List<LayerArg> layersArgs = new ArrayList<>();
        Optional<LayerArg> layerArg;
        while ((layerArg = readLayerArgs(br)).isPresent()) {
            layersArgs.add(layerArg.get());
        }
        return createNetwork(layersArgs);
    }

    private static BufferedReader getBufferedReader(ReadableByteChannel ch) {
        InputStream is = Channels.newInputStream(ch);
        InputStreamReader isr = new InputStreamReader(is, UTF_8);
        return new BufferedReader(isr);
    }

    private static Optional<LayerArg> readLayerArgs(BufferedReader br) throws IOException {
        float[][] weights = readWeightMatrix(br);  // weights between i-th and (i+1) layer
        if (weights.length == 0) {
            return Optional.empty();
        }
        String layerIndexAndFunc = br.readLine(); // activation function for (i+1) layer
        String func = layerIndexAndFunc.split(":", 2)[1].trim();
        ActivationFunc aFunc = ActivationFuncSerializer.deserialize(func);
        LayerArg arg = new LayerArg(aFunc, weights); // (i+1) layer arg
        return Optional.of(arg);
    }

    private static float[][] readWeightMatrix(BufferedReader br) throws IOException {
        String line;
        List<float[]> weights = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            if (line.isEmpty()) {
                break; // end of matrix
            }
            float[] row = parseRow(line);
            weights.add(row);
        }
        return convertToArray(weights);
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

    private static float[][] convertToArray(List<float[]> list) {
        float[][] result = new float[list.size()][];
        int i = 0;
        for (float[] row : list) {
            result[i++] = row;
        }
        return result;
    }

    private static Network createNetwork(List<LayerArg> layerArgs) {
        Assert.isTrue(!layerArgs.isEmpty(), "Minimum 2 layers expected");
        List<Layer> layers = new ArrayList<>();
        int sensorLayerNodeCount = layerArgs.getFirst().weights.length;
        SensorLayer sensorLayer = SensorLayer.of(sensorLayerNodeCount);
        layers.add(sensorLayer);
        for (LayerArg arg : layerArgs) {
            float[][] transposedWeight = Matrix.transpose(arg.weights);
            ResponseLayer layer = ResponseLayer.of(transposedWeight, layers.getLast(), arg.func);
            layers.add(layer);
        }
        return Network.of(layers);
    }

    private record LayerArg(ActivationFunc func, float[][] weights) {
    }
}
