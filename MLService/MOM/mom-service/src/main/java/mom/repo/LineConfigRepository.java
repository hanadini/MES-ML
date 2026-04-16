package mom.repo;

import mom.domain.LineConfig;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface LineConfigRepository extends JpaRepository<LineConfig, Long> {

    Optional<LineConfig> findByLineIdAndRecipeAndShift(
            String lineId,
            String recipe,
            String shift
    );

}