package ai.neuromachines.network.layer;

final class LayerHelper {

    static float[][] copy(float[][] a) {
        float[][] r = new float[a.length][];
        for (int i = 0; i < a.length; i++) {
            r[i] = a[i].clone();
        }
        return r;
    }
}
