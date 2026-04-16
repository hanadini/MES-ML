package mom.domain;

import jakarta.persistence.*;

@Entity
@Table(name = "line_config")
public class LineConfig {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "line_id", nullable = false)
    private String lineId;

    @Column(name = "recipe")
    private String recipe;

    @Column(name = "shift")
    private String shift;

    @Column(name = "warning_threshold", nullable = false)
    private double warningThreshold;

    @Column(name = "high_threshold", nullable = false)
    private double highThreshold;

    @Column(name = "dq_min", nullable = false)
    private double dqMin;

    public LineConfig() {}

    public LineConfig(String lineId,
                      String recipe,
                      String shift,
                      double warningThreshold,
                      double highThreshold,
                      double dqMin) {
        this.lineId = lineId;
        this.recipe = recipe;
        this.shift = shift;
        this.warningThreshold = warningThreshold;
        this.highThreshold = highThreshold;
        this.dqMin = dqMin;
    }

    public String getLineId() { return lineId; }
    public String getRecipe() { return recipe; }
    public String getShift() { return shift; }

    public double getWarningThreshold() { return warningThreshold; }
    public double getHighThreshold() { return highThreshold; }
    public double getDqMin() { return dqMin; }
}