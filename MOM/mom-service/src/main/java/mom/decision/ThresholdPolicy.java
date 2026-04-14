package mom.decision;

public interface ThresholdPolicy {
    Thresholds thresholdsFor(String lineId, String shift, String recipe);
}
