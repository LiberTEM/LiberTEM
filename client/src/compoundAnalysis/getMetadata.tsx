import { AnalysisTypes } from "../messages";
import { CompoundAnalysisMetadata, CompoundAnalysisMetadataItem } from "./types";
// keyof typeof: https://stackoverflow.com/a/42623905/540644
export const getMetadata = (typeName: keyof typeof AnalysisTypes): CompoundAnalysisMetadataItem => {
    const type: AnalysisTypes = AnalysisTypes[typeName];
    return CompoundAnalysisMetadata[type];
};
