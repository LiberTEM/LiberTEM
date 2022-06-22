import * as React from "react";
import { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Button, Checkbox, Dropdown, DropdownProps, Form, Header, Icon, IconProps, Input, List, Modal, Popup } from "semantic-ui-react";
import { defaultDebounce, getEnumValues } from "../../helpers";
import { getApiBasePath } from "../../helpers/apiHelpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisDetails, AnalysisTypes, CenterOfMassParams } from "../../messages";
import { RootReducer } from "../../store";
import { composeHandles } from "../../widgets/compose";
import { cbToRadius, inRectConstraint, keepOnCY, riConstraint, roConstraints } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import { DraggableHandle } from "../../widgets/DraggableHandle";
import Ring from "../../widgets/Ring";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import { haveDisplayResult } from "../helpers";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoCol from "./layouts/AnalysisLayoutTwoCol";
import Toolbar from "./Toolbar";

export enum CoMMaskShapes {
    DISK = "DISK",
    RING = "RING",
}

export const MaskShapeMetadata: { [s: string]: { [s: string]: string } } = {
    [CoMMaskShapes.DISK]: {
        label: "Disk cut-off",
    },
    [CoMMaskShapes.RING]: {
        label: "Annular CoM",
    }
}
const maskShapeKeys = getEnumValues(CoMMaskShapes);
const maskShapeOptions = maskShapeKeys.map(t => ({
    text: MaskShapeMetadata[CoMMaskShapes[t]].label,
    value: CoMMaskShapes[t],
}));

interface MaskShapeSelectorProps {
    selectedShape: CoMMaskShapes,
    handleChange: (e: React.SyntheticEvent, data: DropdownProps) => void,
}

type GuessResponse = {
    status: "error";
    message: string;
} | {
    status: "ok";
    guess: {
        cx: number;
        cy: number;
        scan_rotation: number;
        flip_y: boolean;
    }
}


const MaskShapeSelector: React.FC<MaskShapeSelectorProps> = ({ selectedShape, handleChange }) => (
        <>
            CoM mask shape:{' '}
            <Dropdown inline options={maskShapeOptions}
            value={selectedShape}
            onChange={handleChange}
            />
        </>
    )

const CenterOfMassAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);
    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [r, setR] = useState(minLength / 4);
    const [flip_y, setFlipY] = useState(false);
    const [scan_rotation, setScanRotation] = useState("0.0");
    const [ri, setRI] = useState(minLength / 8);
    const [maskShape, setMaskShape] = useState(CoMMaskShapes.DISK)
    const [guessing, setGuessing] = useState(false);

    const dispatch = useDispatch();

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const riHandle = {
        x: cx - ri,
        y: cy,
    }

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });
    const handleRChange = defaultDebounce(setR);
    const handleRIChange = defaultDebounce(setRI);

    let rConstraint = keepOnCY(cy);
    if (maskShape === CoMMaskShapes.RING) {
        rConstraint = roConstraints(riHandle.x, cy);
    }

    let frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={rConstraint} />
    </>);

    let frameViewWidgets = (<></>);

    if (maskShape === CoMMaskShapes.DISK) {
        frameViewWidgets = (
            <Disk cx={cx} cy={cy} r={r}
                imageWidth={imageWidth} />
        );
    } else if(maskShape === CoMMaskShapes.RING) {
        frameViewWidgets = (
            <Ring cx={cx} cy={cy} ro={r} ri={ri}
                imageWidth={imageWidth} />
        );
        frameViewHandles = composeHandles(frameViewHandles, (handleDragStart, handleDrop)  => (
            <>
                <DraggableHandle x={riHandle.x} y={riHandle.y}
                    imageWidth={imageWidth}
                    onDragMove={cbToRadius(cx, cy, handleRIChange)}
                    parentOnDragStart={handleDragStart}
                    parentOnDrop={handleDrop}
                    constraint={riConstraint(rHandle.x, cy)} />
            </>
        ));
    }

    const {
        frameViewTitle, frameModeSelector,
        handles: resultHandles, widgets: resultWidgets
    } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        compoundAnalysisId: compoundAnalysis.compoundAnalysis,
        doAutoStart: compoundAnalysis.doAutoStart,
    })

    const subtitle = <>{frameViewTitle} Disk: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), r={r.toFixed(2)}</>;

    let parsedScanRotation: number =  parseFloat(scan_rotation);
    if (!parsedScanRotation) {
        parsedScanRotation = 0.0;
    }

    const getDetails: (() => AnalysisDetails) = () => {
        const parameters: CenterOfMassParams = {
            shape: "com",
            cx,
            cy,
            r,
            flip_y,
            scan_rotation: parsedScanRotation,
        };
        if (maskShape === CoMMaskShapes.RING) {
            parameters.ri = ri;
        }
        return {
            analysisType: AnalysisTypes.CENTER_OF_MASS,
            parameters,
        };
    }

    const runAnalysis = () => {
        dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, getDetails()));
    };

    React.useEffect(() => {
        dispatch(compoundAnalysisActions.Actions.setParams(
            compoundAnalysis, 1, getDetails()
        ));
    }, [cx, cy, flip_y, scan_rotation, r, ri, maskShape]);

    const analyses = useSelector((state: RootReducer) => state.analyses)
    const jobs = useSelector((state: RootReducer) => state.jobs)

    const haveResult = haveDisplayResult(
        compoundAnalysis,
        analyses,
        jobs,
        [1],
    );

    // NOTE: haveResult is not a dependency here, as we don't want to re-run directly
    // after the results have become available.
    React.useEffect(() => {
        if (haveResult) {
            runAnalysis();
        }
    }, [flip_y, scan_rotation]);

    const updateFlipY = (e: React.ChangeEvent<HTMLInputElement>, { checked }: { checked: boolean }) => {
        setFlipY(checked);
    };

    const updateScanRotation = (e: React.ChangeEvent<HTMLInputElement>, { value }: { value: string }) => {
        if (value === "-") {
            setScanRotation("-");
        }
        setScanRotation(value);
    };

    const guessParameters = (ev: React.MouseEvent) => {
        setGuessing(true);
        const basePath = getApiBasePath();
        const url = `${basePath}compoundAnalyses/${compoundAnalysis.compoundAnalysis}/rpc/guess_parameters/`;

        fetch(url, {
            method: 'PUT',
            body: JSON.stringify({}),
            credentials: "same-origin",
        }).then(req => req.json()).then((json) => {
            setGuessing(false);
            const response = json as GuessResponse;
            if(response.status === "ok") {
                setFlipY(response.guess.flip_y);
                setCx(response.guess.cx);
                setCy(response.guess.cy);
                setScanRotation(response.guess.scan_rotation.toString());
            } else {
                // eslint-disable-next-line no-console
                console.error(response.message);
            }
        }).catch(e => {
            setGuessing(false);
            // eslint-disable-next-line no-console
            console.error(e)
        });
        ev.preventDefault();
    }

    const guessIconProps: IconProps = guessing ? { name: 'cog', loading: true } : { name: 'question' }

    const toolbar = (
        <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[1]} extra={
            <Button icon onClick={guessParameters} disabled={guessing}>
                <Icon {...guessIconProps} />
                Guess parameters
            </Button>
        }/>
    );


    // TODO: debounce parameters
    const comParams = (
        <>
            <Header>
                <Modal trigger={
                    <Header.Content>
                        Parameters
                        {' '}
                        <Icon name="info circle" size="small" link />
                    </Header.Content>
                }>
                    <Popup.Header>CoM / first moment parameters</Popup.Header>
                    <Popup.Content>
                        <Header>CoM mask shape</Header>
                        <p>
                            Select a shape that will be used to mask out the data:
                        </p>
                        <ul>
                            <li><em>Annular CoM</em>: calculate the center of mass in a selected ring</li>
                            <li><em>Disk cut-off</em>: calculate the center of mass in a selected disk</li>
                        </ul>
                        <Header>Flip in y direction</Header>
                        <p>
                            Flip the Y coordinate. Some detectors, for example Quantum
                            Detectors Merlin, may have pixel (0, 0) at the lower
                            left corner. This has to be corrected to get the sign of
                            the y shift as well as curl and divergence right.
                        </p>
                        <Header>Rotation between scan and detector</Header>
                        <p>
                            The optics of an electron microscope can rotate the
                            image. Furthermore, scan generators may allow
                            scanning in arbitrary directions. This means that
                            the x and y coordinates of the detector image are
                            usually not parallel to the x and y scan
                            coordinates. For interpretation of center of mass
                            shifts, however, the shift vector in detector
                            coordinates has to be put in relation to the
                            position on the sample. This parameter can be used
                            to rotate the detector coordinates to match the scan
                            coordinate system. A positive value rotates the
                            displacement vector clock-wise. That means if the
                            detector seems rotated to the right relative to the
                            scan, this value should be negative to counteract
                            this rotation.
                        </p>
                        <p>
                            Use either the numeric input or the slider to adjust
                            the rotation angle.
                        </p>
                    </Popup.Content>
                </Modal>
            </Header>
            <Form>
                <List relaxed="very">
                    <List.Item>
                        <List.Content>
                            <MaskShapeSelector selectedShape={maskShape} handleChange={(e, data) => {
                                if (data.value === CoMMaskShapes.RING && ri >= r){
                                    // Fixes r being set smaller than ri when the ri handle is hidden
                                    // and the constraint is therefore no longer applied
                                    // Only possible when switching to RING from DISK
                                    // Otherwise maintains previous memory behaviour of r and ri
                                    setRI(r * 0.5)
                                }
                                setMaskShape(data.value as CoMMaskShapes)
                            }} />
                        </List.Content>
                    </List.Item>
                    <List.Item>
                        <List.Content>
                            <Form.Field control={Checkbox} label="Flip in y direction" checked={flip_y} onChange={updateFlipY} />
                        </List.Content>
                    </List.Item>
                    <List.Item>
                        <List.Content>
                            <Form.Field type="number" control={Input} label="Rotation between scan and detector (deg)" value={scan_rotation} onChange={updateScanRotation} />
                            <Form.Field type="range" min="-180" max="180" step="0.1" control={Input} value={scan_rotation} onChange={updateScanRotation} />
                        </List.Content>
                    </List.Item>
                </List>
            </Form>
        </>
    );

    return (
        <AnalysisLayoutTwoCol
            title="CoM / first moment analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                    analysisIndex={0} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            right={<>
                <ResultList
                    analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                    extraWidgets={resultWidgets}
                />
            </>}
            toolbar={toolbar}
            params={comParams}
        />
    );
}

export default CenterOfMassAnalysis;
