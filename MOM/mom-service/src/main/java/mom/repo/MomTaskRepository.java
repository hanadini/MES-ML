package mom.repo;

import mom.domain.MomTask;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MomTaskRepository extends JpaRepository<MomTask, Long> {
    List<MomTask> findByStatus(String status);
    List<MomTask> findByLineId(String lineId);
    List<MomTask> findByOrderId(String orderId);
}
