package ai.neuromachines.file;

import ai.neuromachines.network.Network;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;

import static java.nio.charset.StandardCharsets.UTF_8;

@RequiredArgsConstructor(staticName = "of")
class NetworkReadableByteChannel implements ReadableByteChannel {
    private final Network network;
    @Getter
    private boolean open = true;
    private int currentLayer = -1;
    private ByteBuffer buffer = ByteBuffer.allocate(0);

    // TODO write activation function
    @Override
    public int read(ByteBuffer dst) {
        if (!dst.hasRemaining()) {
            return 0;
        } else if (!buffer.hasRemaining()) {
            // read next layer weights
            currentLayer++;
            if (currentLayer >= (network.layersCount() - 1)) {
                return -1;  // no more layers
            }
            float[][] weights = network.weights(currentLayer);
            buffer = printMatrixToUtf8String(weights);
        }
        int cnt = dst.remaining();
        int limit = buffer.limit();
        int tmpLimit = Math.min(limit, buffer.position() + cnt);  // copy no more than cnt bytes
        buffer.limit(tmpLimit);  // set tmp limit for BufferOverflowException preventing
        dst.put(buffer);
        int transferCnt = cnt - dst.remaining();
        buffer.limit(limit); // restore limit
        return transferCnt;
    }

    private ByteBuffer printMatrixToUtf8String(float[][] weights) {
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

    @Override
    public void close() {
        open = false;
    }
}
