package mom.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.OffsetDateTime;

@Entity
@Table(name = "mom_event")
public class MomEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "event_type", nullable = false)
    private String eventType;

    @Column(name = "line_id", nullable = false)
    private String lineId;

    @Column(name = "order_id", nullable = false)
    private String orderId;

    @Column(name = "risk_level", nullable = false)
    private String riskLevel;

    @Column(name = "probability", nullable = false)
    private double probability;

    @Column(name = "event_time", nullable = false)
    private OffsetDateTime eventTime;

    public MomEvent() {
    }

    public MomEvent(String eventType, String lineId, String orderId, String riskLevel, double probability, OffsetDateTime eventTime) {
        this.eventType = eventType;
        this.lineId = lineId;
        this.orderId = orderId;
        this.riskLevel = riskLevel;
        this.probability = probability;
        this.eventTime = eventTime;
    }

    public Long getId() {
        return id;
    }

    public String getEventType() {
        return eventType;
    }

    public void setEventType(String eventType) {
        this.eventType = eventType;
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

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public double getProbability() {
        return probability;
    }

    public void setProbability(double probability) {
        this.probability = probability;
    }

    public OffsetDateTime getEventTime() {
        return eventTime;
    }

    public void setEventTime(OffsetDateTime eventTime) {
        this.eventTime = eventTime;
    }
}