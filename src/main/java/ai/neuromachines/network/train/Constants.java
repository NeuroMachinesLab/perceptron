package ai.neuromachines.network.train;

/**
 * @see <a href="https://en.wikipedia.org/wiki/Multilayer_perceptron">Multilayer Perceptron</a>
 * @see <a href="https://en.wikipedia.org/wiki/Learning_rate">Learning Rate</a>
 */
public final class Constants {
    /**
     * Коэффициент, определяющий скорость изменения весов
     */
    static float LEARNING_RATE = 0.01f;

    public static void learningRate(float value) {
        LEARNING_RATE = value;
    }
}
