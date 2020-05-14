import * as React from "react";
import { Field } from "formik";
import { Button, Form, FormFieldProps } from "semantic-ui-react";

interface ScanSizeProps {
    value: string,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const ScanSize: React.FC<ScanSizeProps> = ({ value, setFieldValue }) => {

    const [scanSize, setScanSize] = React.useState(value.split(",").length>=2? value.split(","): ["",""]);

    const scanRefsArray = React.useRef<FormFieldProps[]>([]);

    const onScanSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const idx = parseInt(e.target.name.split("_")[2], 10);
      scanSize[idx] = e.target.value.toString();
      setScanSize(scanSize);
      setFieldValue("scan_size", scanSize.toString());
    };

    const onCommaPress = (e: KeyboardEvent) => {
        const idx = parseInt((e.target as HTMLInputElement).name.split("_")[2], 10);
        if(e.keyCode === 188){
            idx===(scanSize.length-1)? newScanDim() : scanRefsArray.current[idx+1].focus();
        }
    }

    const newScanDim = () => {
      scanSize.push("");
      setScanSize(scanSize);
      setFieldValue("scan_size", scanSize.toString());
    }

    React.useEffect(() => {
      if(scanSize.length > 2) {
        scanRefsArray.current[scanSize.length-1].focus();
      }
    }, [scanSize.length]);

    const delScanDim = () => {
      if(scanSize.length > 2) {
        scanSize.pop();
        setScanSize(scanSize);
        setFieldValue("scan_size", scanSize.toString());
      }
    }

    return (
      <>
        <Form.Group>
          {scanSize.map((val, idx) => <Form.Field key={idx} width={2}><Field name={"scan_size_" + idx} id={"id_scan_size_" + idx} type="number" value={val} innerRef={(ref:FormFieldProps) => { scanRefsArray.current[idx] = ref; }} onChange={onScanSizeChange} onKeyDown={onCommaPress} /></Form.Field>)}
          <Form.Field>
            <Button onClick={newScanDim} type="button" icon="add" title="Add dimension" basic={true} color="blue" />
            <Button onClick={delScanDim} type="button" icon="minus" title="Remove dimension" basic={true} color="grey" />
          </Form.Field>
        </Form.Group>
      </>
    );
}

export default ScanSize;
