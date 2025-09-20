package ai.neuromachines;

public class Assert {

    public static void isTrue(boolean expression, String message) {
        if (!expression) {
            throw new IllegalArgumentException(message);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T isInstanceOf(Object obj, Class<? extends T> type) {
        if (type.isInstance(obj)) {
            return (T) obj;
        }
        throw new IllegalArgumentException(type.getSimpleName() + " is expected, got " + obj.getClass().getSimpleName());
    }
}
