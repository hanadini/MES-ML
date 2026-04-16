package mom.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "risk_state")
public class RiskStateEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "line_id", nullable = false)
    private String lineId;

    @Column(name = "order_id", nullable = false)
    private String orderId;

    @Column(name = "current_state", nullable = false)
    private String currentState;

    @Column(name = "warn_count", nullable = false)
    private int warnCount;

    @Column(name = "high_count", nullable = false)
    private int highCount;

    public RiskStateEntity() {
    }

    public RiskStateEntity(String lineId, String orderId, String currentState, int warnCount, int highCount) {
        this.lineId = lineId;
        this.orderId = orderId;
        this.currentState = currentState;
        this.warnCount = warnCount;
        this.highCount = highCount;
    }

    public Long getId() {
        return id;
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

    public String getCurrentState() {
        return currentState;
    }

    public void setCurrentState(String currentState) {
        this.currentState = currentState;
    }

    public int getWarnCount() {
        return warnCount;
    }

    public void setWarnCount(int warnCount) {
        this.warnCount = warnCount;
    }

    public int getHighCount() {
        return highCount;
    }

    public void setHighCount(int highCount) {
        this.highCount = highCount;
    }

}
