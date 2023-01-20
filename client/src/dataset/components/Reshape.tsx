import { ErrorMessage } from "formik";
import * as React from "react";
import { Form, Input } from "semantic-ui-react";
import { framesInfoAfterOffsetCorrection, productOfShapeInCommaSeparatedString } from "../helpers";
import { ShapeLengths } from "../../messages";
import TupleInput from "./TupleInput";

interface ReshapeProps {
    navShape: string,
    sigShape: string,
    syncOffset: number,
    imageCount?: number,
    hideInfo?: boolean,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
    setFieldTouched: (field: string, shouldValidate?: boolean) => void,
}

const Reshape: React.FC<ReshapeProps> = ({
    navShape, sigShape, syncOffset, imageCount=0, hideInfo=false, setFieldValue, setFieldTouched,
}) => {
    const navShapeProduct = productOfShapeInCommaSeparatedString(navShape);

    const { framesSkippedStart, framesIgnoredEnd, framesInsertedStart, framesInsertedEnd } = framesInfoAfterOffsetCorrection(syncOffset, navShapeProduct, imageCount);
    
    const handleOffsetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { value } = e.target;
        let parsed_value: number | string = parseInt(value.toString(), 10)
        if (isNaN(parsed_value)){
            parsed_value = value.toString()
        }
        setFieldValue("sync_offset", parsed_value);
    };

    return (
        <div style={{paddingBottom: 5}}>
            <Form.Field>
                <label htmlFor="id_nav_shape_0">Navigation shape (H, W):</label>
                <div style={{ color: 'red'}}><ErrorMessage name="nav_shape" /></div>
                <TupleInput value={navShape} minLen={ShapeLengths.NAV_SHAPE_MIN_LENGTH} maxLen={ShapeLengths.NAV_SHAPE_MAX_LENGTH} fieldName="nav_shape" setFieldValue={setFieldValue} setFieldTouched={setFieldTouched} />
                <label htmlFor="id_sig_shape_0">Signal shape (H, W):</label>
                <div style={{ color: 'red'}}><ErrorMessage name="sig_shape" /></div>
                <TupleInput value={sigShape} minLen={ShapeLengths.SIG_SHAPE_MIN_LENGTH} maxLen={ShapeLengths.SIG_SHAPE_MAX_LENGTH} fieldName="sig_shape" setFieldValue={setFieldValue} setFieldTouched={setFieldTouched} />
            </Form.Field>
            <Form.Field width={4}>
                <label htmlFor="id_sync_offset">Sync Offset (frames):</label>
                <div style={{ color: 'red'}}><ErrorMessage name="sync_offset" /></div>
                <Input type="number" required name="sync_offset" id="id_sync_offset" value={syncOffset} onChange={handleOffsetChange} />
            </Form.Field>
            <Form.Field hidden={hideInfo}>
                <label>Number of frames skipped at the beginning: {framesSkippedStart}</label>
                <label>Number of blank frames inserted at the beginning: {framesInsertedStart}</label>
                <label>Number of frames ignored at the end: {framesIgnoredEnd}</label>
                <label>Number of blank frames inserted at the end: {framesInsertedEnd}</label>
            </Form.Field>
        </div>
    );
}

export default Reshape;
