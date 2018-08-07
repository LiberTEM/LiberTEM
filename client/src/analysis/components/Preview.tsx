import * as React from "react";
import { getPickFrameURL, getPreviewURL } from "../../dataset/api";
import { DatasetState } from "../../messages";
import { AnalysisState } from "../types";

export interface PreviewProps {
    analysis: AnalysisState,
    dataset: DatasetState,
}


const Preview: React.SFC<PreviewProps> = ({ analysis, dataset }) => {
    const { shape } = dataset.params;

    const imageWidth = shape[3];
    const imageHeight = shape[2];

    let previewURL;

    if (analysis.preview.mode === "AVERAGE") {
        previewURL = getPreviewURL(dataset);
    } else if (analysis.preview.mode === "PICK") {
        if (analysis.preview.pick !== undefined) {
            previewURL = getPickFrameURL(dataset, analysis.preview.pick.x, analysis.preview.pick.y);
        }
    }
    if (previewURL) {
        return <image xlinkHref={previewURL} width={imageWidth} height={imageHeight} style={{ imageRendering: "pixelated" }} />
    }
    return null;
}

export default Preview;