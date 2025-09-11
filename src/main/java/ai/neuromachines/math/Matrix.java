package ai.neuromachines.math;

import ai.neuromachines.Assert;

import java.util.function.Function;

public final class Matrix {

    public static float[] multiply(float[][] weight, float[] vector) {
        int rows = weight.length;
        int cols = vector.length;
        float[] result = new float[rows];
        for (int i = 0; i < rows; i++) {
            float[] weightRow = weight[i];
            Assert.isTrue(weightRow.length == cols, "Weight cols cnt is not equals input rows cnt");
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += weightRow[j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    public static float[] applyFunc(float[] vector, Function<Float, Float> func) {
        int cnt = vector.length;
        float[] result = new float[cnt];
        for (int i = 0; i < cnt; i++) {
            result[i] = func.apply(vector[i]);
        }
        return result;
    }

    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        if (rows == 0) {
            return matrix;
        }
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            float[] row = matrix[i];
            Assert.isTrue(row.length == cols, "Argument is not matrix, columns count is not constant");
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
}
