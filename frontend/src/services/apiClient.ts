type ApiErrorEnvelope = {
  detail?: unknown;
  error?: {
    code?: string;
    status?: number;
    request_id?: string;
    timestamp?: string;
  };
};

export class ApiRequestError extends Error {
  readonly status: number | null;
  readonly requestId: string | null;
  readonly code: string | null;

  constructor(message: string, status: number | null = null, requestId: string | null = null,
    code: string | null = null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.requestId = requestId;
    this.code = code;
  }
}

const validationMessage = (detail: unknown) => {
  if (!Array.isArray(detail)) return null;
  return detail.some(item => typeof item === "object" && item !== null)
    ? "입력한 값을 다시 확인해 주세요."
    : null;
};

const withRequestId = (message: string, requestId: string | null) =>
  requestId ? `${message} · 문의번호: ${requestId}` : message;

async function apiError(response: Response, fallbackMessage: string): Promise<ApiRequestError> {
  const rawBody: unknown = await response.json().catch(() => ({}));
  const body = typeof rawBody === "object" && rawBody !== null
    ? rawBody as ApiErrorEnvelope
    : {};
  const requestId = body.error?.request_id ?? response.headers.get("X-Request-ID");
  const detail = typeof body.detail === "string"
    ? body.detail
    : validationMessage(body.detail);
  const message = detail ?? (response.status === 404
    ? "요청한 정보를 찾을 수 없습니다."
    : response.status === 403
      ? "이 요청을 실행할 권한이 없습니다."
      : fallbackMessage);
  return new ApiRequestError(
    withRequestId(message, requestId),
    response.status,
    requestId,
    body.error?.code ?? null,
  );
}

export async function apiRequest<T>(url: string, options: {
  signal?: AbortSignal;
  fallbackMessage: string;
}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { signal: options.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiRequestError("서버에 연결할 수 없습니다. 네트워크 상태를 확인한 뒤 다시 시도해 주세요.");
  }

  if (!response.ok) throw await apiError(response, options.fallbackMessage);
  try {
    return await response.json() as T;
  } catch {
    const requestId = response.headers.get("X-Request-ID");
    throw new ApiRequestError(
      withRequestId("서버 응답을 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.", requestId),
      response.status,
      requestId,
    );
  }
}

export const userErrorMessage = (error: unknown, fallbackMessage: string) =>
  error instanceof Error && error.message.trim() ? error.message : fallbackMessage;
