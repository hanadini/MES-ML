package mom.service;

import mom.domain.MomTask;
import mom.repo.MomTaskRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TaskQueryService {

    private final MomTaskRepository momTaskRepository;

    public TaskQueryService(MomTaskRepository momTaskRepository) {
        this.momTaskRepository = momTaskRepository;
    }

    public List<MomTask> getTasks(String status, String lineId, String orderId) {
        if (status != null && !status.isBlank()) {
            return momTaskRepository.findByStatus(status);
        }
        if (lineId != null && !lineId.isBlank()) {
            return momTaskRepository.findByLineId(lineId);
        }
        if (orderId != null && !orderId.isBlank()) {
            return momTaskRepository.findByOrderId(orderId);
        }
        return momTaskRepository.findAll();
    }
}
