package mom.api;

import mom.domain.MomEvent;
import mom.service.EventQueryService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/events")
public class EventController {

    private final EventQueryService eventQueryService;

    public EventController(EventQueryService eventQueryService) {
        this.eventQueryService = eventQueryService;
    }

    @GetMapping
    public List<MomEvent> getEvents(
            @RequestParam(required = false) String lineId,
            @RequestParam(required = false) String orderId
    ) {
        return eventQueryService.getEvents(lineId, orderId);
    }
}
