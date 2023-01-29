from typing import List, Optional
from src.training_type import TrainingType
from src.class_mappers import AbstractClassMapper
from src.structs import UnmergedMaskData, MergedMaskData, MaskRepresentantion, MaskColor

class MaskDataMerger:
    def __init__(self, args) -> None:
        self.binary = args.training_type == TrainingType.BINARY_SEG
        self.allow_ambigous_mappings = args.training_type == TrainingType.MULTILABEL_CLASSIFICATION

    def merge(self, masks_data: List[UnmergedMaskData], mapper: AbstractClassMapper) -> Optional[List[MergedMaskData]]:
        unmerged_mask_data_with_mapped_classes = self.__map_classes(masks_data, mapper)
        all_mapped_classes = {mask_data.class_name for mask_data in unmerged_mask_data_with_mapped_classes}

        merged_mask_data = []
        for processed_class in all_mapped_classes:
            if self.binary and not mapper.is_positive(processed_class):
                merged_mask_data.append(MergedMaskData(processed_class, [MaskRepresentantion.of_color(MaskColor.BLACK)]))
            else:
                masks_paths_of_class = [mask_data.mask_path for mask_data in unmerged_mask_data_with_mapped_classes if mask_data.class_name == processed_class]
                merged_mask = self.__merge_masks_data(masks_paths_of_class, processed_class)
                merged_mask_data.append(merged_mask)

        return merged_mask_data if len(merged_mask_data) > 0 else None


    def __map_classes(self, masks_data: List[UnmergedMaskData], mapper: AbstractClassMapper) -> List[UnmergedMaskData]:
        if not self.allow_ambigous_mappings:
            masks_data = self.__drop_records_where_single_mask_is_mapped_to_multiple_sets_of_classes(masks_data, mapper)

        result = []
        for mask_data in masks_data:
            mapped_classes = mapper.map(mask_data.class_name)
            for mapped_class in mapped_classes:
                result.append(UnmergedMaskData(mapped_class, mask_data.mask_path))
            
        return result

    def __drop_records_where_single_mask_is_mapped_to_multiple_sets_of_classes(self, masks_data: List[UnmergedMaskData], mapper: AbstractClassMapper) -> List[UnmergedMaskData]:
        mask_paths_to_drop = set()
        mapped_classes_per_mask = {}
        for mask_data in masks_data:
            class_name = mask_data.class_name
            mask_path = mask_data.mask_path
            mapped_classes = set(mapper.map(class_name))
            if len(mapped_classes) == 0:
                continue

            if mask_path in mapped_classes_per_mask:
                if mapped_classes_per_mask[mask_path] != mapped_classes:
                    print(f"[WARN] Skipping mask at location {mask_path}. It is mapped to multiple different sets of classes. Conflict {mapped_classes_per_mask[mask_path]} vs {mapped_classes}.")
                    mask_paths_to_drop.add(mask_path)
            else:
                mapped_classes_per_mask[mask_path] = mapped_classes

        return [mask_data for mask_data in masks_data if mask_data.mask_path not in mask_paths_to_drop]
        

    def __merge_masks_data(self, masks_paths: List[str], class_name: str) -> MergedMaskData:
        masks_repr = [MaskRepresentantion.of_path(mask_path) for mask_path in masks_paths]
        return MergedMaskData(class_name, masks_repr)

