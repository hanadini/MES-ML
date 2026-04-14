package mom.decision;

import java.util.Map;

public class InMemoryThresholdPolicy implements ThresholdPolicy{
    private final Map<String, Thresholds> map;
    public InMemoryThresholdPolicy(Map<String, Thresholds> map){
        this.map = map;
    }

    @Override
    public Thresholds thresholdsFor(String lineId, String shift, String recipe){
        String key = lineId + "|" + shift + "|" + recipe;
        Thresholds t = map.get(key);
        if (t != null) return t;

        // fallback keys (more generic)
        t = map.get(lineId + "|" + shift + "|*");
        if (t != null) return t;

        t = map.get(lineId + "|*|*");
        if (t != null) return t;

        // default safe values
        return new Thresholds(0.65, 0.75);
    }
}
