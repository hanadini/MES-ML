package mom.service;

import mom.domain.MomEvent;
import mom.repo.MomEventRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EventQueryService {

    private final MomEventRepository momEventRepository;

    public EventQueryService(MomEventRepository momEventRepository) {
        this.momEventRepository = momEventRepository;
    }

    public List<MomEvent> getEvents(String lineId, String orderId) {
        if (lineId != null && !lineId.isBlank()) {
            return momEventRepository.findByLineId(lineId);
        }
        if (orderId != null && !orderId.isBlank()) {
            return momEventRepository.findByOrderId(orderId);
        }
        return momEventRepository.findAll();
    }
}
