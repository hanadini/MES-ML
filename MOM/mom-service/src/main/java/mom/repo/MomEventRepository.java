package mom.repo;

import mom.domain.MomEvent;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MomEventRepository extends JpaRepository<MomEvent, Long> {
    List<MomEvent> findByLineId(String lineId);
    List<MomEvent> findByOrderId(String orderId);
}
