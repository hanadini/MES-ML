package mom.repo;

import mom.domain.RiskStateEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface RiskStateRepository extends JpaRepository<RiskStateEntity, Long> {
    Optional<RiskStateEntity> findByLineIdAndOrderId(String lineId, String orderId);
}