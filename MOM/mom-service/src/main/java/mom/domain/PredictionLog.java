package mom.domain;

import jakarta.persistence.*;

import java.time.OffsetDateTime;

@Entity
@Table(name = "prediction_log")
public class PredictionLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "request_id", nullable = false)
    private String requestId;

    @Column(name = "line_id", nullable = false)
    private String lineId;

    @Column(name = "order_id", nullable = false)
    private String orderId;

    @Column(name = "probability", nullable = false)
    private double probability;

    @Column(name = "risk_level", nullable = false)
    private String riskLevel;

    @Column(name = "applied_state", nullable = false)
    private String appliedState;

    @Column(name = "applied_threshold", nullable = false)
    private double appliedThreshold;

    @Column(name = "data_quality_score", nullable = false)
    private double dataQualityScore;

    @Column(name = "fallback_used", nullable = false)
    private boolean fallbackUsed;

    @Column(name = "model_version")
    private String modelVersion;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    public PredictionLog() {
    }

    public PredictionLog(String requestId,
                         String lineId,
                         String orderId,
                         double probability,
                         String riskLevel,
                         String appliedState,
                         double appliedThreshold,
                         double dataQualityScore,
                         boolean fallbackUsed,
                         String modelVersion,
                         OffsetDateTime createdAt) {
        this.requestId = requestId;
        this.lineId = lineId;
        this.orderId = orderId;
        this.probability = probability;
        this.riskLevel = riskLevel;
        this.appliedState = appliedState;
        this.appliedThreshold = appliedThreshold;
        this.dataQualityScore = dataQualityScore;
        this.fallbackUsed = fallbackUsed;
        this.modelVersion = modelVersion;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public String getLineId() {
        return lineId;
    }

    public void setLineId(String lineId) {
        this.lineId = lineId;
    }

    public String getOrderId() {
        return orderId;
    }

    public void setOrderId(String orderId) {
        this.orderId = orderId;
    }

    public double getProbability() {
        return probability;
    }

    public void setProbability(double probability) {
        this.probability = probability;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getAppliedState() {
        return appliedState;
    }

    public void setAppliedState(String appliedState) {
        this.appliedState = appliedState;
    }

    public double getAppliedThreshold() {
        return appliedThreshold;
    }

    public void setAppliedThreshold(double appliedThreshold) {
        this.appliedThreshold = appliedThreshold;
    }

    public double getDataQualityScore() {
        return dataQualityScore;
    }

    public void setDataQualityScore(double dataQualityScore) {
        this.dataQualityScore = dataQualityScore;
    }

    public boolean isFallbackUsed() {
        return fallbackUsed;
    }

    public void setFallbackUsed(boolean fallbackUsed) {
        this.fallbackUsed = fallbackUsed;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }
}