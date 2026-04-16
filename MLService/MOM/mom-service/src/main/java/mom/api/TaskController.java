package mom.api;

import mom.domain.MomTask;
import mom.service.TaskQueryService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/tasks")
public class TaskController {

    private final TaskQueryService taskQueryService;

    public TaskController(TaskQueryService taskQueryService) {
        this.taskQueryService = taskQueryService;
    }

    @GetMapping
    public List<MomTask> getTasks(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String lineId,
            @RequestParam(required = false) String orderId
    ) {
        return taskQueryService.getTasks(status, lineId, orderId);
    }
}