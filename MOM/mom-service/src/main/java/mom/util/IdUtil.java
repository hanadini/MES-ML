package mom.util;

import java.util.UUID;

public final class IdUtil {

    private IdUtil() {
    }

    public static String newRequestId() {
        return UUID.randomUUID().toString();
    }

    public static String newCorrelationId() {
        return UUID.randomUUID().toString();
    }

    public static String newShortTaskCode() {
        return "TASK-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }
}